"""
MedGuard — Fixed Trainer
========================
Key fixes over original:
  1. Class weights for ALL 3 heads (DDI, NER, Severity)
  2. NER padding tokens masked with -100 so CrossEntropyLoss ignores them
  3. Severity label distribution printed so you can verify Head 3 data
  4. Full F1 evaluation per class for all 3 heads after every epoch
  5. Saves best model by macro-F1 (not just loss)
  6. Correct __main__ paths for your project layout

Run from backend/:
    python -m app.models.trainer
"""

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
from sklearn.metrics import f1_score, classification_report
import numpy as np

from app.models.medguard_model import MedGuardModel, DDI_LABELS
from app.data.preprocessor import load_ddi_corpus, DDISentence

LABEL2IDX = {'false': 0, 'mechanism': 1, 'effect': 2, 'advise': 3, 'int': 4}
IDX2LABEL  = {v: k for k, v in LABEL2IDX.items()}

NER_LABEL2IDX = {'O': 0, 'B-DRUG': 1, 'I-DRUG': 2}
SEV_LABEL2IDX = {'safe': 0, 'caution': 1, 'warning': 2, 'danger': 3}
SEV_IDX2LABEL = {v: k for k, v in SEV_LABEL2IDX.items()}

# Padding positions in NER labels are marked -100 so loss ignores them
NER_PAD_LABEL = -100


# ── Severity lookup ───────────────────────────────────────────────────────────

def load_severity_lookup(db_path: str) -> Dict:
    """Load all DrugBank severity pairs into memory."""
    lookup = {}
    if not os.path.exists(db_path):
        print(f"  ⚠️  drugbank.db not found at {db_path} — severity labels will all be 0")
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
            v = min(int(severity), 3)
            lookup[(drug_a, drug_b)] = v
            lookup[(drug_b, drug_a)] = v
        print(f"  Loaded {len(lookup)//2} severity pairs into memory")
    except Exception as e:
        print(f"  Severity lookup error: {e}")
    return lookup


# ── Dataset ───────────────────────────────────────────────────────────────────

class DDIDataset(Dataset):
    """
    PyTorch Dataset for DDI Corpus 2013 — all 3 MTL heads.

    NER fix: padding/special tokens are labelled NER_PAD_LABEL (-100)
    so CrossEntropyLoss(ignore_index=-100) skips them correctly.
    Previously they were labelled 0 ("O"), which meant the loss was
    dominated by non-drug tokens and the head never learned drug spans.
    """

    def __init__(
        self,
        sentences: List[DDISentence],
        tokenizer,
        max_length: int = 128,
        severity_lookup: Dict = None
    ):
        self.samples         = []
        self.tokenizer       = tokenizer
        self.max_length      = max_length
        self.severity_lookup = severity_lookup or {}
        self._build_samples(sentences)

    def _build_samples(self, sentences: List[DDISentence]):
        for sent in sentences:
            if len(sent.interactions) == 0:
                continue

            encoding = self.tokenizer(
                sent.text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            input_ids      = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()

            # ── NER labels ────────────────────────────────────────────────────
            # Start everything as NER_PAD_LABEL (-100).
            # Real tokens get 0 (O), B-DRUG tokens get 1, I-DRUG tokens get 2.
            # Special/padding tokens stay at -100 and are ignored by the loss.
            ner_labels = [NER_PAD_LABEL] * self.max_length

            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == 0 and token_end == 0:
                    continue          # [CLS], [SEP], [PAD] → stay -100
                ner_labels[idx] = 0  # real token → default "O"

            for entity in sent.entities:
                first_token = True
                for idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == 0 and token_end == 0:
                        continue
                    if token_start >= entity.start and token_end <= entity.end + 1:
                        ner_labels[idx] = 1 if first_token else 2
                        first_token = False

            for interaction in sent.interactions:
                ddi_label      = LABEL2IDX.get(interaction.get('type', 'false'), 0)
                severity_label = self._get_severity(sent, interaction)

                self.samples.append({
                    'input_ids':      input_ids,
                    'attention_mask': attention_mask,
                    'ner_labels':     torch.tensor(ner_labels,      dtype=torch.long),
                    'label':          torch.tensor(ddi_label,        dtype=torch.long),
                    'severity_label': torch.tensor(severity_label,   dtype=torch.long),
                    'text':           sent.text,
                })

    def _get_severity(self, sent: DDISentence, interaction: Dict) -> int:
        e1_id = interaction.get('e1', '')
        e2_id = interaction.get('e2', '')
        drug_a = drug_b = None
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


# ── Class weight helpers ──────────────────────────────────────────────────────

def compute_ddi_class_weights(sentences: List[DDISentence]) -> torch.Tensor:
    """Balanced class weights for Head 2 (DDI interaction type)."""
    labels = []
    for sent in sentences:
        for interaction in sent.interactions:
            labels.append(LABEL2IDX.get(interaction.get('type', 'false'), 0))

    labels   = np.array(labels)
    classes  = np.unique(labels)
    weights  = compute_class_weight('balanced', classes=classes, y=labels)

    weight_tensor = torch.ones(len(LABEL2IDX))
    for cls, w in zip(classes, weights):
        weight_tensor[cls] = w

    print(f"  DDI class weights:      {weight_tensor.tolist()}")
    return weight_tensor


def compute_ner_class_weights(dataset: DDIDataset) -> torch.Tensor:
    """
    Balanced class weights for Head 1 (NER).
    Only counts real tokens (ignores -100 padding).
    """
    all_labels = []
    for sample in dataset.samples:
        lbls = sample['ner_labels'].tolist()
        all_labels.extend([l for l in lbls if l != NER_PAD_LABEL])

    all_labels = np.array(all_labels)
    classes    = np.unique(all_labels)
    weights    = compute_class_weight('balanced', classes=classes, y=all_labels)

    # 3 NER classes: O=0, B-DRUG=1, I-DRUG=2
    weight_tensor = torch.ones(3)
    for cls, w in zip(classes, weights):
        weight_tensor[int(cls)] = w

    print(f"  NER class weights:      {weight_tensor.tolist()}")
    return weight_tensor


def compute_severity_class_weights(dataset: DDIDataset) -> torch.Tensor:
    """Balanced class weights for Head 3 (Severity)."""
    labels = [s['severity_label'].item() for s in dataset.samples]

    # Print distribution so you can see if severity data is present
    from collections import Counter
    dist = Counter(labels)
    print(f"  Severity distribution:  {dict(sorted(dist.items()))}")

    labels   = np.array(labels)
    classes  = np.unique(labels)
    weights  = compute_class_weight('balanced', classes=classes, y=labels)

    weight_tensor = torch.ones(4)
    for cls, w in zip(classes, weights):
        weight_tensor[int(cls)] = w

    print(f"  Severity class weights: {weight_tensor.tolist()}")
    return weight_tensor


# ── Training ──────────────────────────────────────────────────────────────────

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
    """Train one epoch — all 3 heads with gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids       = batch['input_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        ddi_labels      = batch['label'].to(device)
        ner_labels      = batch['ner_labels'].to(device)
        severity_labels = batch['severity_label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Head 2: DDI interaction loss (weighted)
        loss_ddi = criterion_ddi(outputs['interaction_logits'], ddi_labels)

        # Head 1: NER loss — ignore_index=-100 skips padding/special tokens
        ner_logits = outputs['ner_logits']
        batch_size, seq_len, num_ner = ner_logits.shape
        loss_ner = criterion_ner(
            ner_logits.view(-1, num_ner),
            ner_labels.view(-1)
        )

        # Head 3: Severity loss (weighted)
        loss_severity = criterion_severity(
            outputs['severity_logits'], severity_labels
        )

        # Combined MTL loss
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
                f"  Step {step:4d}/{len(dataloader)} — "
                f"loss={loss.item()*accumulation_steps:.4f}  "
                f"ddi={loss_ddi.item():.4f}  "
                f"ner={loss_ner.item():.4f}  "
                f"sev={loss_severity.item():.4f}"
            )

    return total_loss / len(dataloader)


# ── Evaluation with F1 ────────────────────────────────────────────────────────

def evaluate(
    model,
    dataloader,
    criterion_ddi,
    device
) -> Dict:
    """
    Full evaluation — loss + accuracy + macro-F1 for all 3 heads.
    Returns a dict with all metrics.
    """
    model.eval()

    total_loss = 0.0

    # Head 2 — DDI
    ddi_preds_all  = []
    ddi_labels_all = []

    # Head 1 — NER (only real tokens, not padding)
    ner_preds_all  = []
    ner_labels_all = []

    # Head 3 — Severity
    sev_preds_all  = []
    sev_labels_all = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ddi_labels     = batch['label'].to(device)
            ner_labels     = batch['ner_labels'].to(device)
            sev_labels     = batch['severity_label'].to(device)

            outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
            loss     = criterion_ddi(outputs['interaction_logits'], ddi_labels)
            total_loss += loss.item()

            # Head 2
            ddi_preds = outputs['interaction_logits'].argmax(dim=-1)
            ddi_preds_all.extend(ddi_preds.cpu().tolist())
            ddi_labels_all.extend(ddi_labels.cpu().tolist())

            # Head 1 — flatten and filter out -100 padding
            ner_logits = outputs['ner_logits']
            ner_preds  = ner_logits.argmax(dim=-1)   # (batch, seq)
            flat_preds  = ner_preds.view(-1).cpu().tolist()
            flat_labels = ner_labels.view(-1).cpu().tolist()
            for p, l in zip(flat_preds, flat_labels):
                if l != NER_PAD_LABEL:
                    ner_preds_all.append(p)
                    ner_labels_all.append(l)

            # Head 3
            sev_preds = outputs['severity_logits'].argmax(dim=-1)
            sev_preds_all.extend(sev_preds.cpu().tolist())
            sev_labels_all.extend(sev_labels.cpu().tolist())

    # ── Metrics ──────────────────────────────────────────────────────────────
    avg_loss = total_loss / len(dataloader)

    # DDI
    ddi_acc     = sum(p == l for p, l in zip(ddi_preds_all, ddi_labels_all)) / len(ddi_labels_all)
    ddi_f1_macro = f1_score(ddi_labels_all, ddi_preds_all, average='macro', zero_division=0)
    ddi_f1_per   = f1_score(ddi_labels_all, ddi_preds_all, average=None,    zero_division=0, labels=list(range(5)))

    # NER
    ner_acc      = sum(p == l for p, l in zip(ner_preds_all, ner_labels_all)) / max(len(ner_labels_all), 1)
    ner_f1_macro = f1_score(ner_labels_all, ner_preds_all, average='macro', zero_division=0)
    ner_f1_per   = f1_score(ner_labels_all, ner_preds_all, average=None,    zero_division=0, labels=[0, 1, 2])

    # Severity
    sev_acc      = sum(p == l for p, l in zip(sev_preds_all, sev_labels_all)) / max(len(sev_labels_all), 1)
    sev_f1_macro = f1_score(sev_labels_all, sev_preds_all, average='macro', zero_division=0)
    sev_f1_per   = f1_score(sev_labels_all, sev_preds_all, average=None,    zero_division=0, labels=[0, 1, 2, 3])

    return {
        'loss':          avg_loss,
        # DDI
        'ddi_acc':       ddi_acc,
        'ddi_f1_macro':  ddi_f1_macro,
        'ddi_f1_per':    ddi_f1_per,
        # NER
        'ner_acc':       ner_acc,
        'ner_f1_macro':  ner_f1_macro,
        'ner_f1_per':    ner_f1_per,
        # Severity
        'sev_acc':       sev_acc,
        'sev_f1_macro':  sev_f1_macro,
        'sev_f1_per':    sev_f1_per,
    }


def print_metrics(metrics: Dict, prefix: str = "Val"):
    """Pretty-print all metrics."""
    ddi_names = ['false', 'mechanism', 'effect', 'advise', 'int']
    ner_names = ['O', 'B-DRUG', 'I-DRUG']
    sev_names = ['safe', 'caution', 'warning', 'danger']

    print(f"\n  {'─'*52}")
    print(f"  {prefix} Results")
    print(f"  {'─'*52}")
    print(f"  Loss:              {metrics['loss']:.4f}")
    print(f"\n  HEAD 2 — DDI Interaction")
    print(f"  Accuracy:          {metrics['ddi_acc']:.4f}")
    print(f"  Macro F1:          {metrics['ddi_f1_macro']:.4f}  ← key metric")
    for name, f1 in zip(ddi_names, metrics['ddi_f1_per']):
        bar = '█' * int(f1 * 20)
        print(f"    {name:<12} F1={f1:.4f}  {bar}")
    print(f"\n  HEAD 1 — NER Drug Detection")
    print(f"  Accuracy:          {metrics['ner_acc']:.4f}")
    print(f"  Macro F1:          {metrics['ner_f1_macro']:.4f}  ← key metric")
    for name, f1 in zip(ner_names, metrics['ner_f1_per']):
        bar = '█' * int(f1 * 20)
        print(f"    {name:<12} F1={f1:.4f}  {bar}")
    print(f"\n  HEAD 3 — Severity")
    print(f"  Accuracy:          {metrics['sev_acc']:.4f}")
    print(f"  Macro F1:          {metrics['sev_f1_macro']:.4f}  ← key metric")
    for name, f1 in zip(sev_names, metrics['sev_f1_per']):
        bar = '█' * int(f1 * 20)
        print(f"    {name:<12} F1={f1:.4f}  {bar}")
    print(f"  {'─'*52}")


# ── Main training function ────────────────────────────────────────────────────

def train(
    data_dir:          str,
    output_dir:        str,
    model_name:        str   = "emilyalsentzer/Bio_ClinicalBERT",
    num_epochs:        int   = 5,
    batch_size:        int   = 4,
    learning_rate:     float = 2e-5,
    accumulation_steps:int   = 4,
    val_split:         float = 0.15,
):
    """Full training pipeline — all 3 MTL heads with F1 evaluation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    corpus_dir = os.path.join(data_dir, 'DDICorpus')
    db_path    = os.path.join(data_dir, 'drugbank.db')

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading DDI Corpus...")
    train_sentences, test_sentences = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(
        train_sentences, test_size=val_split, random_state=42
    )
    print(f"Train: {len(train_sents)} | Val: {len(val_sents)} | Test: {len(test_sentences)}")

    print("\nLoading severity data...")
    severity_lookup = load_severity_lookup(db_path)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\nBuilding datasets...")
    train_dataset = DDIDataset(train_sents, tokenizer, severity_lookup=severity_lookup)
    val_dataset   = DDIDataset(val_sents,   tokenizer, severity_lookup=severity_lookup)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Class weights for all 3 heads ─────────────────────────────────────────
    print("\nComputing class weights...")
    ddi_weights = compute_ddi_class_weights(train_sents).to(device)
    ner_weights = compute_ner_class_weights(train_dataset).to(device)
    sev_weights = compute_severity_class_weights(train_dataset).to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading MedGuard model...")
    model = MedGuardModel(model_name=model_name).to(device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps  = len(train_loader) * num_epochs // accumulation_steps
    warmup_steps = total_steps // 10
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── Loss functions — ALL 3 heads now have class weights ───────────────────
    criterion_ddi      = nn.CrossEntropyLoss(weight=ddi_weights)
    criterion_ner      = nn.CrossEntropyLoss(weight=ner_weights, ignore_index=NER_PAD_LABEL)
    criterion_severity = nn.CrossEntropyLoss(weight=sev_weights)

    best_ddi_f1 = 0.0
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*54)
    print("Training all 3 MTL heads:")
    print("  Head 1: NER        — drug entity recognition")
    print("  Head 2: Interaction — DDI Corpus 2013 (5 classes)")
    print("  Head 3: Severity   — DrugBank severity (4 classes)")
    print("  Saving best model by DDI macro-F1")
    print("="*54)

    for epoch in range(num_epochs):
        print(f"\n{'='*54}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*54}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion_ddi, criterion_ner, criterion_severity,
            device, accumulation_steps
        )
        print(f"\n  Train Loss: {train_loss:.4f}")

        metrics = evaluate(model, val_loader, criterion_ddi, device)
        print_metrics(metrics, prefix=f"Epoch {epoch+1} Val")

        # Save best model by DDI macro-F1 (not loss — loss doesn't reflect class balance)
        if metrics['ddi_f1_macro'] > best_ddi_f1:
            best_ddi_f1 = metrics['ddi_f1_macro']
            save_path = os.path.join(output_dir, 'best_model_3heads.pt')
            torch.save(model.state_dict(), save_path)
            print(f"\n  ✅ Best model saved (DDI macro-F1={best_ddi_f1:.4f}) → {save_path}")

    print("\n" + "="*54)
    print("Training complete!")
    print(f"Best DDI macro-F1: {best_ddi_f1:.4f}")
    print("="*54)
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run from backend/:  python -m app.models.trainer
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR   = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

    print(f"DATA_DIR   : {DATA_DIR}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR}")

    train(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        num_epochs=5,
        batch_size=4,
        learning_rate=2e-5,
        accumulation_steps=4,
    )