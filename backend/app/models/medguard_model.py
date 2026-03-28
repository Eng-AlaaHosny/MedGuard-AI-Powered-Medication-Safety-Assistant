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
    3: 'B-DOSAGE',
    4: 'B-CONDITION'
}

SEVERITY_LABELS = {
    0: 'safe',
    1: 'caution',
    2: 'warning',
    3: 'danger'
}

SEVERITY_COLORS = {
    'safe': '#28a745',
    'caution': '#ffc107',
    'warning': '#fd7e14',
    'danger': '#dc3545'
}


class MedGuardModel(nn.Module):

    def __init__(
        self,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        num_ner_labels=5,
        num_ddi_labels=5,
        num_severity_labels=4,
        kg_embedding_dim=128,
        dropout=0.1
    ):
        super(MedGuardModel, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.ner_head = nn.Linear(self.hidden_size, num_ner_labels)

        self.pair_embedding = nn.Embedding(1, self.hidden_size)
        pair_input_dim = self.hidden_size * 3

        self.kg_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + kg_embedding_dim, self.hidden_size),
            nn.GELU()
        )

        self.pair_projection = nn.Sequential(
            nn.Linear(pair_input_dim, self.hidden_size),
            nn.GELU()
        )

        self.interaction_head = nn.Linear(self.hidden_size, num_ddi_labels)

        self.severity_head = nn.Linear(self.hidden_size, num_severity_labels)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        token_representations = outputs.last_hidden_state
        cls_representation = outputs.last_hidden_state[:, 0, :]
        return token_representations, cls_representation

    def get_entity_representation(self, token_repr, entity_spans):
        if entity_spans is None or len(entity_spans) == 0:
            return token_repr[:, 0, :]
        start, end = entity_spans[0]
        entity_tokens = token_repr[:, start:end+1, :]
        return entity_tokens.mean(dim=1)

    def fuse_kg_embedding(self, bert_repr, kg_embedding):
        if kg_embedding is None:
            batch_size = bert_repr.size(0)
            kg_embedding = torch.zeros(batch_size, 128, device=bert_repr.device)
        combined = torch.cat([bert_repr, kg_embedding], dim=-1)
        return self.kg_fusion(combined)

    def forward(
        self,
        input_ids,
        attention_mask,
        drug_a_spans=None,
        drug_b_spans=None,
        kg_embedding_a=None,
        kg_embedding_b=None
    ):
        token_repr, cls_repr = self.encode(input_ids, attention_mask)
        token_repr = self.dropout(token_repr)
        cls_repr = self.dropout(cls_repr)

        ner_logits = self.ner_head(token_repr)

        h_drug_a = self.get_entity_representation(token_repr, drug_a_spans)
        h_drug_b = self.get_entity_representation(token_repr, drug_b_spans)

        h_drug_a = self.fuse_kg_embedding(h_drug_a, kg_embedding_a)
        h_drug_b = self.fuse_kg_embedding(h_drug_b, kg_embedding_b)

        pair_idx = torch.zeros(cls_repr.size(0), dtype=torch.long, device=cls_repr.device)
        h_pair = self.pair_embedding(pair_idx)

        drug_pair_repr = torch.cat([h_drug_a, h_drug_b, h_pair], dim=-1)
        drug_pair_repr = self.pair_projection(drug_pair_repr)

        interaction_logits = self.interaction_head(drug_pair_repr)
        severity_logits = self.severity_head(drug_pair_repr)

        return {
            'ner_logits': ner_logits,
            'interaction_logits': interaction_logits,
            'severity_logits': severity_logits
        }


def load_tokenizer(model_name="emilyalsentzer/Bio_ClinicalBERT"):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name="emilyalsentzer/Bio_ClinicalBERT"):
    model = MedGuardModel(model_name=model_name)
    return model


if __name__ == "__main__":
    print("Loading Bio_ClinicalBERT tokenizer and model...")
    print("This will download ~436MB on first run - please wait...")

    tokenizer = load_tokenizer()
    model = load_model()
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

    print("\n Model loaded successfully!")
    print(f"NER logits shape:          {outputs['ner_logits'].shape}")
    print(f"Interaction logits shape:  {outputs['interaction_logits'].shape}")
    print(f"Severity logits shape:     {outputs['severity_logits'].shape}")
    print("\nAll 3 MTL heads working correctly!")
