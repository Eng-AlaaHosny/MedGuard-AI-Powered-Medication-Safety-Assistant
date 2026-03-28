import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Dict, List

NS = {'db': 'http://www.drugbank.ca'}


def map_severity(description: str) -> int:
    """
    Map interaction description to severity score.
    0 = safe, 1 = caution, 2 = warning, 3 = danger
    Blueprint Section 3.1
    """
    if not description:
        return 0
    desc_lower = description.lower()
    if any(word in desc_lower for word in [
        'major', 'contraindicated', 'severe', 'serious', 'fatal',
        'dangerous', 'life-threatening', 'toxic', 'toxicity'
    ]):
        return 3
    elif any(word in desc_lower for word in [
        'moderate', 'significant', 'monitor', 'increase', 'decrease',
        'risk', 'bleeding', 'anticoagulant', 'inhibit', 'enhance'
    ]):
        return 2
    elif any(word in desc_lower for word in [
        'minor', 'mild', 'slight', 'may', 'possible', 'potential'
    ]):
        return 1
    return 0


def parse_drugbank_xml(xml_path: str) -> List[Dict]:
    """Parse DrugBank XML and extract drug + interaction data."""
    print(f"Parsing DrugBank XML from {xml_path} ...")
    drugs = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for drug in root.findall('db:drug', NS):
        drugbank_id_el = drug.find('db:drugbank-id[@primary="true"]', NS)
        name_el = drug.find('db:name', NS)
        description_el = drug.find('db:description', NS)
        mechanism_el = drug.find('db:mechanism-of-action', NS)

        drug_id = drugbank_id_el.text if drugbank_id_el is not None else ''
        drug_name = name_el.text if name_el is not None else ''
        description = description_el.text if description_el is not None else ''
        mechanism = mechanism_el.text if mechanism_el is not None else ''

        interactions = []
        interactions_el = drug.find('db:drug-interactions', NS)
        if interactions_el is not None:
            for interaction in interactions_el.findall('db:drug-interaction', NS):
                interacting_id_el = interaction.find('db:drugbank-id', NS)
                interacting_name_el = interaction.find('db:name', NS)
                desc_el = interaction.find('db:description', NS)

                interacting_id = interacting_id_el.text if interacting_id_el is not None else ''
                interacting_name = interacting_name_el.text if interacting_name_el is not None else ''
                desc = desc_el.text if desc_el is not None else ''
                severity = map_severity(desc)

                interactions.append({
                    'drug_b_id': interacting_id,
                    'drug_b_name': interacting_name,
                    'description': desc,
                    'severity': severity
                })

        drugs.append({
            'id': drug_id,
            'name': drug_name,
            'description': description,
            'mechanism': mechanism,
            'interactions': interactions
        })

    print(f"Parsed {len(drugs)} drugs from DrugBank")
    return drugs


def build_sqlite_db(drugs: List[Dict], db_path: str):
    """Insert parsed DrugBank data into SQLite for fast inference lookup."""
    print(f"Building SQLite database at {db_path} ...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS drugs (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        mechanism TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        drug_a_id TEXT,
        drug_b_id TEXT,
        drug_b_name TEXT,
        description TEXT,
        severity INTEGER,
        PRIMARY KEY (drug_a_id, drug_b_id)
    )''')

    for drug in drugs:
        c.execute('INSERT OR REPLACE INTO drugs VALUES (?, ?, ?, ?)',
                  (drug['id'], drug['name'], drug['description'], drug['mechanism']))
        for interaction in drug['interactions']:
            c.execute('INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?)',
                      (drug['id'], interaction['drug_b_id'], interaction['drug_b_name'],
                       interaction['description'], interaction['severity']))

    conn.commit()
    conn.close()
    print("SQLite database built successfully")


def lookup_interaction(drug_a_name: str, drug_b_name: str, db_path: str) -> Dict:
    """Look up interaction between two drugs by name."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''
        SELECT i.description, i.severity, i.drug_b_name
        FROM interactions i
        JOIN drugs da ON i.drug_a_id = da.id
        JOIN drugs db ON i.drug_b_id = db.id
        WHERE (LOWER(da.name) = LOWER(?) AND LOWER(db.name) = LOWER(?))
        OR (LOWER(da.name) = LOWER(?) AND LOWER(db.name) = LOWER(?))
    ''', (drug_a_name, drug_b_name, drug_b_name, drug_a_name))

    row = c.fetchone()
    conn.close()

    if row:
        severity_labels = {0: 'safe', 1: 'caution', 2: 'warning', 3: 'danger'}
        return {
            'found': True,
            'description': row[0],
            'severity': row[1],
            'severity_label': severity_labels.get(row[1], 'unknown')
        }
    return {'found': False}


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, 'drugbank_full.xml', 'full database.xml')
    db_path = os.path.join(base_dir, 'drugbank.db')

    if os.path.exists(xml_path):
        print(f"Found DrugBank XML at: {xml_path}")
        drugs = parse_drugbank_xml(xml_path)
        build_sqlite_db(drugs, db_path)
        print(f"\nDone! Total drugs: {len(drugs)}")
        print(f"SQLite database saved to: {db_path}")
    else:
        print(f"DrugBank XML not found at: {xml_path}")