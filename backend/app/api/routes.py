import os
import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from app.models.medguard_model import (
    MedGuardModel, load_model, load_tokenizer,
    DDI_LABELS, NER_LABELS, SEVERITY_LABELS, SEVERITY_COLORS
)
from app.knowledge_graph.graph_builder import DrugKnowledgeGraph
from app.data.lipinski_processor import LipinskiProcessor

router = APIRouter()

# ── Global singletons (loaded once at startup via lifespan in main.py) ────────
model: Optional[MedGuardModel] = None
tokenizer = None
kg: Optional[DrugKnowledgeGraph] = None
lipinski: Optional[LipinskiProcessor] = None

MAX_LENGTH = 128


# ── Request / Response schemas ────────────────────────────────────────────────

class DDIRequest(BaseModel):
    text: str
    drug_names: Optional[List[str]] = []   # explicit names from UI — used when NER misses


class DetectedEntity(BaseModel):
    text: str
    start: int
    end: int
    label: str                      # B-DRUG / I-DRUG


class DDIResponse(BaseModel):
    text: str
    detected_entities: List[DetectedEntity]
    interaction_type: str
    interaction_type_idx: int
    severity_label: str
    severity_level: int
    severity_color: str
    severity_source: str            # "knowledge_graph" or "model"
    interaction_reason: str         # human-readable reason sentence
    kg_context: Dict
    lipinski_context: Dict
    confidence: Dict


# ── Resource helpers ──────────────────────────────────────────────────────────

def get_resources():
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading — please retry in a moment."
        )
    return model, tokenizer, kg, lipinski


# ── NER decoder ───────────────────────────────────────────────────────────────

def decode_ner(
    text: str,
    ner_logits: torch.Tensor,
    offset_mapping: List[tuple]
) -> List[DetectedEntity]:
    predictions = ner_logits.argmax(dim=-1).squeeze(0).tolist()
    entities: List[DetectedEntity] = []

    current_start: Optional[int] = None
    current_end:   Optional[int] = None
    current_label: Optional[str] = None

    for idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == 0 and token_end == 0:
            if current_start is not None:
                entities.append(DetectedEntity(
                    text=text[current_start:current_end],
                    start=current_start, end=current_end, label=current_label
                ))
                current_start = current_end = current_label = None
            continue

        label_name = NER_LABELS.get(predictions[idx], "O")

        if label_name == "B-DRUG":
            if current_start is not None:
                entities.append(DetectedEntity(
                    text=text[current_start:current_end],
                    start=current_start, end=current_end, label=current_label
                ))
            current_start = token_start
            current_end   = token_end
            current_label = "B-DRUG"
        elif label_name == "I-DRUG" and current_start is not None:
            current_end = token_end
        else:
            if current_start is not None:
                entities.append(DetectedEntity(
                    text=text[current_start:current_end],
                    start=current_start, end=current_end, label=current_label
                ))
                current_start = current_end = current_label = None

    if current_start is not None:
        entities.append(DetectedEntity(
            text=text[current_start:current_end],
            start=current_start, end=current_end, label=current_label
        ))

    seen = set()
    unique: List[DetectedEntity] = []
    for e in entities:
        key = (e.start, e.end)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def synthetic_entities(drug_names: List[str], text: str) -> List[DetectedEntity]:
    """
    When NER detects nothing, create synthetic B-DRUG entities from the
    explicit drug_names passed by the frontend. Finds them in the text
    by simple case-insensitive search.
    """
    entities = []
    for name in drug_names:
        idx = text.lower().find(name.lower())
        if idx != -1:
            entities.append(DetectedEntity(
                text=name, start=idx, end=idx + len(name), label="B-DRUG"
            ))
        else:
            # Not in text at all — still add as a synthetic entity at position 0
            entities.append(DetectedEntity(
                text=name, start=0, end=len(name), label="B-DRUG"
            ))
    return entities


# ── KG context builder ────────────────────────────────────────────────────────

def build_kg_context(
    entities: List[DetectedEntity],
    graph: Optional[DrugKnowledgeGraph]
) -> Dict:
    if graph is None:
        return {"status": "Knowledge Graph not loaded"}

    drug_names = [e.text for e in entities if e.label == "B-DRUG"]

    if len(drug_names) < 2:
        availability = {name: graph.check_drug_available(name) for name in drug_names}
        return {
            "status": "Fewer than 2 drugs detected — pair analysis unavailable",
            "drugs_in_graph": availability
        }

    drug_a, drug_b = drug_names[0], drug_names[1]
    a_available = graph.check_drug_available(drug_a)
    b_available = graph.check_drug_available(drug_b)

    result = {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "drug_a_in_graph": a_available,
        "drug_b_in_graph": b_available,
    }

    if not a_available or not b_available:
        missing = [d for d, av in [(drug_a, a_available), (drug_b, b_available)] if not av]
        result["status"] = f"Data Unavailable — {', '.join(missing)} not in Knowledge Graph"
        result["known_interaction"] = None
    else:
        interaction_info = graph.get_interaction_info(drug_a, drug_b)
        if interaction_info:
            severity_labels = {0: "safe", 1: "caution", 2: "warning", 3: "danger"}
            result["status"] = "Known interaction found in DrugBank KG"
            result["known_interaction"] = {
                "severity": interaction_info.get("severity", 0),
                "severity_label": severity_labels.get(interaction_info.get("severity", 0), "unknown"),
                "description": interaction_info.get("description", ""),
                "interaction_type": interaction_info.get("interaction_type", ""),
            }
        else:
            result["status"] = "Both drugs in graph — no direct interaction edge found"
            result["known_interaction"] = None

    return result


# ── Lipinski context builder ──────────────────────────────────────────────────

def build_lipinski_context(
    entities: List[DetectedEntity],
    lip: Optional[LipinskiProcessor]
) -> Dict:
    if lip is None:
        return {"status": "Lipinski processor not loaded"}

    drug_names = [e.text for e in entities if e.label == "B-DRUG"]
    if not drug_names:
        return {"status": "No drug entities detected"}

    drug_data = {}
    for name in drug_names:
        features = lip.get_features(name)
        if features is not None:
            drug_data[name] = {
                "available": True,
                "molecular_weight": round(float(features[0]), 2),
                "n_hba": int(features[1]),
                "n_hbd": int(features[2]),
                "logp": round(float(features[3]), 2),
                "ro5_fulfilled": bool(features[4]),
            }
        else:
            drug_data[name] = {
                "available": False,
                "status": "Data Unavailable — drug not in Lipinski dataset"
            }
    return {"drugs": drug_data}


# ── KG embedding retrieval ────────────────────────────────────────────────────

def get_kg_embedding_tensor(
    drug_name: str,
    graph: Optional[DrugKnowledgeGraph],
    device: str
) -> Optional[torch.Tensor]:
    if graph is None:
        return None
    embedding = graph.get_drug_embedding(drug_name)
    if embedding is None:
        return None
    return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)


# ── KG severity override ──────────────────────────────────────────────────────

def get_kg_severity(
    drug_names: List[str],
    graph: Optional[DrugKnowledgeGraph]
) -> Optional[int]:
    if graph is None or len(drug_names) < 2:
        return None
    interaction_info = graph.get_interaction_info(drug_names[0], drug_names[1])
    if interaction_info is not None:
        return int(interaction_info.get("severity", 0))
    return None


# ── Interaction reason builder ────────────────────────────────────────────────

DDI_TYPE_DESCRIPTIONS = {
    "false":     "No clinically significant interaction detected between these drugs.",
    "mechanism": "Pharmacokinetic interaction — one drug affects the metabolism or absorption of the other.",
    "effect":    "Pharmacodynamic interaction — the drugs produce additive or opposing effects when combined.",
    "advise":    "Advisory interaction — use with caution and monitor the patient closely.",
    "int":       "Interaction detected — consult clinical guidelines before co-administering.",
}

def build_interaction_reason(
    interaction_type: str,
    kg_context: Dict,
    drug_names: List[str]
) -> str:
    """
    Return a human-readable sentence explaining the model prediction.
    The model interaction type is always the primary message.
    KG description is only appended as extra context when available.
    """
    pair = f"{drug_names[0]} and {drug_names[1]}" if len(drug_names) >= 2 else "These drugs"

    # 1. Model prediction sentence — always shown
    base = DDI_TYPE_DESCRIPTIONS.get(interaction_type, "Interaction detected.")
    reason = f"{pair}: {base}"

    # 2. If KG has a description AND model agrees there is an interaction,
    #    replace with the more informative KG description
    if interaction_type != "false":
        known = kg_context.get("known_interaction")
        if known and known.get("description"):
            desc = known["description"].strip()
            if len(desc) > 200:
                cut = desc[:200].rfind(".")
                desc = desc[:cut + 1] if cut > 100 else desc[:200] + "…"
            reason = desc

    return reason


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=DDIResponse)
async def analyze_interaction(request: DDIRequest):
    m, t, graph, lip = get_resources()
    device = next(m.parameters()).device

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text must not be empty.")

    explicit_drug_names = [d.strip() for d in (request.drug_names or []) if d.strip()]

    # ── Step 1: Tokenize ──────────────────────────────────────────────────────
    encoding = t(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

    # ── Step 2: Quick first-pass NER ─────────────────────────────────────────
    with torch.no_grad():
        first_pass = m(input_ids=input_ids, attention_mask=attention_mask)

    preliminary_entities = decode_ner(text, first_pass["ner_logits"], offset_mapping)
    ner_drug_names = [e.text for e in preliminary_entities if e.label == "B-DRUG"]

    # If NER found nothing, use the explicit names sent by the frontend
    if len(ner_drug_names) < 2 and len(explicit_drug_names) >= 2:
        drug_names_for_kg = explicit_drug_names
    else:
        drug_names_for_kg = ner_drug_names

    # ── Step 3: Retrieve KG embeddings ───────────────────────────────────────
    kg_emb_a = get_kg_embedding_tensor(drug_names_for_kg[0], graph, str(device)) \
        if len(drug_names_for_kg) > 0 else None
    kg_emb_b = get_kg_embedding_tensor(drug_names_for_kg[1], graph, str(device)) \
        if len(drug_names_for_kg) > 1 else None

    # ── Step 4: Full model forward pass ──────────────────────────────────────
    with torch.no_grad():
        outputs = m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kg_embedding_a=kg_emb_a,
            kg_embedding_b=kg_emb_b,
        )

    # ── Step 5: Decode heads ──────────────────────────────────────────────────
    entities = decode_ner(text, outputs["ner_logits"], offset_mapping)

    # If NER still detects nothing, synthesise entities from explicit names
    if not any(e.label == "B-DRUG" for e in entities) and explicit_drug_names:
        entities = synthetic_entities(explicit_drug_names, text)

    interaction_probs = torch.softmax(outputs["interaction_logits"], dim=-1).squeeze(0)
    severity_probs    = torch.softmax(outputs["severity_logits"],    dim=-1).squeeze(0)

    interaction_idx = int(interaction_probs.argmax().item())
    model_severity  = int(severity_probs.argmax().item())
    interaction_type = DDI_LABELS.get(interaction_idx, "false")

    # ── Step 6: Model heads are ALWAYS the source of truth ─────────────────
    # Head 1 (NER) detected the drug entities above.
    # Head 2 (Interaction) decides the interaction type.
    # Head 3 (Severity) decides the severity level.
    # The KG is only used for the description/reason text and as extra context
    # in the UI — it never overrides the model output.
    severity_idx    = model_severity
    severity_source = "model"

    severity_label = SEVERITY_LABELS.get(severity_idx, "safe")
    severity_color = SEVERITY_COLORS.get(severity_label, "#28a745")

    confidence = {
        "interaction": {
            DDI_LABELS[i]: round(float(interaction_probs[i].item()), 4)
            for i in range(len(DDI_LABELS))
        },
        "severity": {
            SEVERITY_LABELS[i]: round(float(severity_probs[i].item()), 4)
            for i in range(len(SEVERITY_LABELS))
        },
    }

    # ── Step 7: Context enrichment ────────────────────────────────────────────
    kg_context       = build_kg_context(entities, graph)
    lipinski_context = build_lipinski_context(entities, lip)

    # ── Step 8: Build reason sentence ────────────────────────────────────────
    interaction_reason = build_interaction_reason(
        interaction_type, kg_context, drug_names_for_kg
    )

    return DDIResponse(
        text=text,
        detected_entities=entities,
        interaction_type=interaction_type,
        interaction_type_idx=interaction_idx,
        severity_label=severity_label,
        severity_level=severity_idx,
        severity_color=severity_color,
        severity_source=severity_source,
        interaction_reason=interaction_reason,
        kg_context=kg_context,
        lipinski_context=lipinski_context,
        confidence=confidence,
    )


# ── Health / status endpoints ─────────────────────────────────────────────────

@router.get("/health")
def health_check():
    return {
        "status": "ready" if model is not None else "loading",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "kg_loaded": kg is not None,
        "lipinski_loaded": lipinski is not None,
    }


@router.get("/drugs")
def list_kg_drugs(limit: int = 50):
    if kg is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not loaded.")
    drugs = list(kg.drug_name_to_id.keys())[:limit]
    return {"count": len(kg.drug_name_to_id), "sample": drugs}