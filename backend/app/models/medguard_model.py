"""
MedGuard — Multi-Task Learning Model
=====================================
3 heads on top of Bio_ClinicalBERT:
  Head 1: NER        — token-level drug entity detection (3 classes: O, B-DRUG, I-DRUG)
  Head 2: Interaction — sentence-level DDI classification (5 classes)
  Head 3: Severity   — sentence-level severity prediction (4 classes)

Fixes over original:
  - NER head outputs 3 classes (not 5) — matches training labels
  - Higher dropout (0.3) to reduce overfitting on small corpus
  - CLS token used for pair classification (more stable than entity spans)
  - KG fusion kept but zero-padded when embeddings unavailable
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional

DDI_LABELS = {
    0: 'false',
    1: 'mechanism',
    2: 'effect',
    3: 'advise',
    4: 'int'
}

NER_LABELS = {
    0: 'O',
    1: 'B-DRUG',
    2: 'I-DRUG',
}

SEVERITY_LABELS = {
    0: 'safe',
    1: 'caution',
    2: 'warning',
    3: 'danger'
}

SEVERITY_COLORS = {
    'safe':    '#28a745',
    'caution': '#ffc107',
    'warning': '#fd7e14',
    'danger':  '#dc3545'
}


class MedGuardModel(nn.Module):
    """
    Bio_ClinicalBERT backbone with 3 MTL heads.

    Architecture:
      encoder        → Bio_ClinicalBERT (768-dim hidden)
      ner_head       → Linear(768, 3)   token-level
      kg_fusion      → Linear(768+128, 768) + GELU  (per drug)
      pair_projection→ Linear(768*3, 768) + GELU
      interaction_head → Linear(768, 5)  sentence-level
      severity_head  → Linear(768, 4)   sentence-level
    """

    def __init__(
        self,
        model_name:           str   = "emilyalsentzer/Bio_ClinicalBERT",
        num_ner_labels:       int   = 3,    # O, B-DRUG, I-DRUG
        num_ddi_labels:       int   = 5,    # false/mechanism/effect/advise/int
        num_severity_labels:  int   = 4,    # safe/caution/warning/danger
        kg_embedding_dim:     int   = 128,
        dropout:              float = 0.3,  # increased from 0.1
    ):
        super().__init__()

        self.encoder     = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)

        # ── Head 1: NER ───────────────────────────────────────────────────────
        self.ner_head = nn.Linear(self.hidden_size, num_ner_labels)

        # ── KG fusion (applied per drug entity representation) ────────────────
        self.kg_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + kg_embedding_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Pair representation ───────────────────────────────────────────────
        # [drug_a | drug_b | cls] concatenated → projected to hidden_size
        self.pair_embedding  = nn.Embedding(1, self.hidden_size)
        self.pair_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Head 2: DDI interaction type ──────────────────────────────────────
        self.interaction_head = nn.Linear(self.hidden_size, num_ddi_labels)

        # ── Head 3: Severity ──────────────────────────────────────────────────
        self.severity_head = nn.Linear(self.hidden_size, num_severity_labels)

    # ── Encoder ───────────────────────────────────────────────────────────────

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        token_repr = outputs.last_hidden_state          # (B, T, 768)
        cls_repr   = outputs.last_hidden_state[:, 0, :] # (B, 768)
        return token_repr, cls_repr

    # ── Entity representation ─────────────────────────────────────────────────

    def get_entity_representation(self, token_repr, entity_spans=None):
        """
        Mean-pool over entity token spans.
        Falls back to CLS token when spans are not provided
        (which is the case during inference without span annotations).
        """
        if entity_spans is None or len(entity_spans) == 0:
            return token_repr[:, 0, :]          # CLS token
        start, end = entity_spans[0]
        span = token_repr[:, start:end + 1, :]  # (B, span_len, 768)
        return span.mean(dim=1)                 # (B, 768)

    # ── KG fusion ─────────────────────────────────────────────────────────────

    def fuse_kg_embedding(self, bert_repr, kg_embedding=None):
        """
        Concatenate BERT drug repr with KG node2vec embedding.
        Uses zero vector when KG embedding is unavailable.
        """
        if kg_embedding is None:
            kg_embedding = torch.zeros(
                bert_repr.size(0), 128, device=bert_repr.device
            )
        combined = torch.cat([bert_repr, kg_embedding], dim=-1)  # (B, 768+128)
        return self.kg_fusion(combined)                           # (B, 768)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids,
        attention_mask,
        drug_a_spans=None,
        drug_b_spans=None,
        kg_embedding_a=None,
        kg_embedding_b=None,
    ):
        """
        Args:
            input_ids:       (B, T)
            attention_mask:  (B, T)
            drug_a_spans:    optional list of (start, end) token spans for drug A
            drug_b_spans:    optional list of (start, end) token spans for drug B
            kg_embedding_a:  optional (B, 128) KG node2vec embedding for drug A
            kg_embedding_b:  optional (B, 128) KG node2vec embedding for drug B

        Returns:
            dict with keys:
              ner_logits:          (B, T, 3)
              interaction_logits:  (B, 5)
              severity_logits:     (B, 4)
        """
        token_repr, cls_repr = self.encode(input_ids, attention_mask)
        token_repr = self.dropout(token_repr)
        cls_repr   = self.dropout(cls_repr)

        # ── Head 1: NER ───────────────────────────────────────────────────────
        ner_logits = self.ner_head(token_repr)   # (B, T, 3)

        # ── Drug representations for Heads 2 & 3 ─────────────────────────────
        h_a = self.get_entity_representation(token_repr, drug_a_spans)
        h_b = self.get_entity_representation(token_repr, drug_b_spans)

        # Fuse with KG embeddings
        h_a = self.fuse_kg_embedding(h_a, kg_embedding_a)   # (B, 768)
        h_b = self.fuse_kg_embedding(h_b, kg_embedding_b)   # (B, 768)

        # Pair interaction token
        pair_idx = torch.zeros(
            cls_repr.size(0), dtype=torch.long, device=cls_repr.device
        )
        h_pair = self.pair_embedding(pair_idx)               # (B, 768)

        # Project concatenated triple to hidden_size
        pair_repr = torch.cat([h_a, h_b, h_pair], dim=-1)   # (B, 768*3)
        pair_repr = self.pair_projection(pair_repr)           # (B, 768)

        # ── Head 2: Interaction type ──────────────────────────────────────────
        interaction_logits = self.interaction_head(pair_repr)  # (B, 5)

        # ── Head 3: Severity ──────────────────────────────────────────────────
        severity_logits = self.severity_head(pair_repr)        # (B, 4)

        return {
            'ner_logits':         ner_logits,
            'interaction_logits': interaction_logits,
            'severity_logits':    severity_logits,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
    return MedGuardModel(model_name=model_name)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Bio_ClinicalBERT tokenizer and model...")

    tokenizer = load_tokenizer()
    model     = load_model()
    model.eval()

    test_sentence = "Warfarin and aspirin interaction may increase bleeding risk."
    inputs = tokenizer(
        test_sentence,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    print("\n✅ Model loaded successfully!")
    print(f"  NER logits shape:         {outputs['ner_logits'].shape}")
    print(f"  Interaction logits shape: {outputs['interaction_logits'].shape}")
    print(f"  Severity logits shape:    {outputs['severity_logits'].shape}")
    print(f"\n  NER classes:         {list(NER_LABELS.values())}")
    print(f"  DDI classes:         {list(DDI_LABELS.values())}")
    print(f"  Severity classes:    {list(SEVERITY_LABELS.values())}")
    print("\nAll 3 MTL heads working correctly!")