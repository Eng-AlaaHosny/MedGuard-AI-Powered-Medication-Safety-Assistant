import os
import sqlite3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from app.models.medguard_model import MedGuardModel, DDI_LABELS
from app.data.preprocessor import load_ddi_corpus, DDISentence

LABEL2IDX = {'false': 0, 'mechanism': 1, 'effect': 2, 'advise': 3, 'int': 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


def load_severity_lookup(db_path: str) -> Dict:
    """
    Load ALL severity data into memory at once.
    Much faster than querying DB for each sample.
    """
    lookup = {}
    if not os.path.exists(db_path):
        return lookup
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            SELECT LOWER(da.name), LOWER(db.name), i.severity
            FROM interactions i
            JOIN drugs da ON i.drug_a_id = da.id
            JOIN drugs db ON i.drug_b_id = db.id
        ''')
        rows = c.fetchall()
        conn.close()
        for drug_a, drug_b, severity in rows:
            lookup[(drug_a, drug_b)] = min(int(severity), 3)
            lookup[(drug_b, drug_a)] = min(int(severity), 3)
        print(f"Loaded {len(lookup)} severity pairs into memory")
    except Exception as e:
        print(f"Severity lookup error: {e}")
    return lookup


class DDIDataset(Dataset):
    """
    PyTorch Dataset for DDI Corpus 2013.
    Trains all 3 heads:
    - Head 1: NER (token-level drug detection)
    - Head 2: Interaction classification
    - Head 3: Severity prediction
    """

    def __init__(
        self,
        sentences: List[DDISentence],
        tokenizer,
        max_length: int = 128,
        severity_lookup: Dict = None
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.severity_lookup = severity_lookup or {}
        self._build_samples(sentences)

    def _build_samples(self, sentences: List[DDISentence]):
        for sent in sentences:
            if len(sent.interactions) == 0:
                continue

            # Tokenize with offset mapping for NER
            encoding = self.tokenizer(
                sent.text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()

            # Build NER labels for Head 1
            ner_labels = [0] * self.max_length
            for entity in sent.entities:
                first_token = True
                for idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == 0 and token_end == 0:
                        continue
                    if token_start >= entity.start and token_end <= entity.end + 1:
                        if first_token:
                            ner_labels[idx] = 1  # B-DRUG
                            first_token = False
                        else:
                            ner_labels[idx] = 2  # I-DRUG

            for interaction in sent.interactions:
                # Head 2: interaction label
                ddi_label = LABEL2IDX.get(
                    interaction.get('type', 'false'), 0
                )

                # Head 3: severity from memory lookup
                severity_label = self._get_severity(sent, interaction)

                self.samples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'ner_labels': torch.tensor(ner_labels, dtype=torch.long),
                    'label': torch.tensor(ddi_label, dtype=torch.long),
                    'severity_label': torch.tensor(severity_label, dtype=torch.long),
                    'text': sent.text
                })

    def _get_severity(self, sent: DDISentence, interaction: Dict) -> int:
        """Get severity from in-memory lookup."""
        e1_id = interaction.get('e1', '')
        e2_id = interaction.get('e2', '')

        drug_a = None
        drug_b = None
        for entity in sent.entities:
            if entity.id == e1_id:
                drug_a = entity.text.lower()
            if entity.id == e2_id:
                drug_b = entity.text.lower()

        if drug_a and drug_b:
            return self.severity_lookup.get((drug_a, drug_b), 0)
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def compute_ddi_class_weights(sentences: List[DDISentence]) -> torch.Tensor:
    """Compute class weights for imbalanced DDI labels."""
    labels = []
    for sent in sentences:
        for interaction in sent.interactions:
            label = LABEL2IDX.get(interaction.get('type', 'false'), 0)
            labels.append(label)

    labels = np.array(labels)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)

    weight_tensor = torch.ones(len(LABEL2IDX))
    for cls, weight in zip(classes, weights):
        weight_tensor[cls] = weight

    print(f"Class weights: {weight_tensor}")
    return weight_tensor


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion_ddi,
    criterion_ner,
    criterion_severity,
    device,
    accumulation_steps=4
) -> float:
    """Train one epoch with all 3 heads."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ddi_labels = batch['label'].to(device)
        ner_labels = batch['ner_labels'].to(device)
        severity_labels = batch['severity_label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Head 2: Interaction loss
        loss_ddi = criterion_ddi(outputs['interaction_logits'], ddi_labels)

        # Head 1: NER loss
        ner_logits = outputs['ner_logits']
        batch_size, seq_len, num_labels = ner_logits.shape
        loss_ner = criterion_ner(
            ner_logits.view(-1, num_labels),
            ner_labels.view(-1)
        )

        # Head 3: Severity loss
        loss_severity = criterion_severity(
            outputs['severity_logits'], severity_labels
        )

        # Combined loss — blueprint Section 3.1
        loss = loss_ddi + 0.3 * loss_ner + 0.3 * loss_severity
        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        if step % 100 == 0:
            print(
                f"  Step {step}/{len(dataloader)} — "
                f"Total: {loss.item()*accumulation_steps:.4f} | "
                f"DDI: {loss_ddi.item():.4f} | "
                f"NER: {loss_ner.item():.4f} | "
                f"Sev: {loss_severity.item():.4f}"
            )

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion_ddi, device) -> Tuple[float, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion_ddi(outputs['interaction_logits'], labels)
            total_loss += loss.item()

            preds = outputs['interaction_logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), accuracy


def train(
    data_dir: str,
    output_dir: str,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    accumulation_steps: int = 4,
    val_split: float = 0.15
):
    """Full training pipeline — all 3 MTL heads."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    corpus_dir = os.path.join(data_dir, 'DDICorpus')
    db_path = os.path.join(data_dir, 'drugbank.db')

    # Load DDI Corpus
    print("Loading DDI Corpus...")
    train_sentences, test_sentences = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(
        train_sentences, test_size=val_split, random_state=42
    )
    print(f"Train: {len(train_sents)} | Val: {len(val_sents)} | Test: {len(test_sentences)}")

    # Load severity lookup into memory ONCE
    print("Loading severity data into memory...")
    severity_lookup = load_severity_lookup(db_path)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build datasets
    print("Building datasets...")
    train_dataset = DDIDataset(
        train_sents, tokenizer, severity_lookup=severity_lookup
    )
    val_dataset = DDIDataset(
        val_sents, tokenizer, severity_lookup=severity_lookup
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Class weights
    class_weights = compute_ddi_class_weights(train_sents).to(device)

    # Load model
    print("Loading MedGuard model...")
    model = MedGuardModel(model_name=model_name).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Loss functions for all 3 heads
    criterion_ddi = nn.CrossEntropyLoss(weight=class_weights)
    criterion_ner = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_severity = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Training all 3 MTL heads:")
    print("  Head 1: NER — drug entity recognition")
    print("  Head 2: Interaction — DDI Corpus 2013")
    print("  Head 3: Severity — DrugBank descriptions")
    print("="*50)

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion_ddi, criterion_ner, criterion_severity,
            device, accumulation_steps
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion_ddi, device)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, 'best_model_3heads.pt')
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved → {save_path}")

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*50)
    return model


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'models', 'checkpoints')

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        num_epochs=5,
        batch_size=4,
        learning_rate=2e-5
    )
