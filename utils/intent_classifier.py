"""
utils/intent_classifier.py — SetFit intent classifier for query routing.

Replaces the centroid-averaging heuristic in agents.py with a proper
few-shot trained classifier (Tunstall et al., 2022 — arxiv:2209.11055).

SetFit fine-tunes a sentence transformer using contrastive pairs from a
small labeled dataset (12-16 examples/class). It outperformed GPT-3 on
few-shot classification at 1/1600th the model size.

6 classes: approval, timeline, person, attachment, status, summary

"summary" was previously handled by a cosine-threshold pre-check in agents.py
because the old model didn't know the class. This version trains summary as a
proper class so the pre-check is no longer needed.

Training:
  Run this file directly once to train and save the model:
      python utils/intent_classifier.py
  The model is saved to data/setfit_intent_model/ and loaded at runtime.
  If the saved model does not exist, falls back to centroid averaging.
"""

from __future__ import annotations
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

MODEL_SAVE_DIR = Path("data/setfit_intent_model")
BASE_MODEL     = "BAAI/bge-small-en-v1.5"

# ──────────────────────────────────────────────────────────────────────
# Labeled training examples — 16 per class (96 total)
#
# Design principles:
#   • "summary" vs "status": summary asks for a COMPREHENSIVE OVERVIEW of
#     the WHOLE thread; status asks about a SPECIFIC OUTCOME or CURRENT STATE.
#   • "person" includes "in this thread" variants to avoid cosine drift toward
#     the summary centroid.
#   • "approval" stays focused on financial/authority-granting decisions.
#   • Each class has hard negatives nearby (the model must learn the boundary).
# ──────────────────────────────────────────────────────────────────────

TRAIN_EXAMPLES = [
    # ── approval ──────────────────────────────────────────────────────
    ("What was the approved budget for the project?",                    "approval"),
    ("Did finance sign off on the vendor contract?",                     "approval"),
    ("How much was approved for the Q4 infrastructure spend?",           "approval"),
    ("Who authorized the payment to the contractor?",                    "approval"),
    ("Was the $2 million capital expenditure approved by the board?",    "approval"),
    ("What is the approved amount for storage vendor fees?",             "approval"),
    ("Did the CFO approve the acquisition terms?",                       "approval"),
    ("What did Ken Lay sign off on last quarter?",                       "approval"),
    ("Was the deal approved before the deadline?",                       "approval"),
    ("What budget line items were rejected by the committee?",           "approval"),
    ("Did the board accept the succession plan recommendation?",         "approval"),
    ("How much did management approve for the energy trading desk?",     "approval"),
    ("Was the merger sanctioned by senior leadership?",                  "approval"),
    ("Who gave the green light for the expansion?",                      "approval"),
    ("What is the authorized spending limit for this project?",          "approval"),
    ("Did the exec committee approve the new hire?",                     "approval"),

    # ── timeline ──────────────────────────────────────────────────────
    ("When was the succession plan announced?",                          "timeline"),
    ("What is the sequence of events in this thread?",                   "timeline"),
    ("When were these emails sent?",                                     "timeline"),
    ("What happened first in this negotiation?",                         "timeline"),
    ("Give me a chronological view of the thread",                       "timeline"),
    ("When did Jeff Skilling officially become CEO?",                    "timeline"),
    ("What was the timeline for the transition?",                        "timeline"),
    ("When did the board meeting take place?",                           "timeline"),
    ("What date did Ken Lay send the announcement?",                     "timeline"),
    ("List the milestones in order",                                     "timeline"),
    ("What was the last email in this chain?",                           "timeline"),
    ("How long did the approval process take?",                          "timeline"),
    ("In what order did these events unfold?",                           "timeline"),
    ("When did the discussions first start?",                            "timeline"),
    ("What was the most recent update?",                                 "timeline"),
    ("What happened between January and March?",                         "timeline"),

    # ── person ────────────────────────────────────────────────────────
    ("Who sent the congratulations email?",                              "person"),
    ("Who are the key people mentioned in this thread?",                 "person"),
    ("Which executive made the announcement?",                           "person"),
    ("Who is Jeff Skilling reporting to?",                               "person"),
    ("Who replied to the succession plan announcement?",                 "person"),
    ("Which person raised concerns about the timeline?",                 "person"),
    ("Who is CC'd on this email chain?",                                 "person"),
    ("Who is the sender of the original message?",                       "person"),
    ("Which Enron employee wrote this?",                                 "person"),
    ("Who is responsible for investor outreach?",                        "person"),
    ("Name the participants in this email thread",                       "person"),
    ("Who forwarded this to the whole company?",                         "person"),
    ("Who are the people involved in this thread?",                      "person"),
    ("Who are the people involved in this discussion?",                  "person"),
    ("Who are the participants in this conversation?",                   "person"),
    ("List all the people who appear in these emails",                   "person"),

    # ── attachment ────────────────────────────────────────────────────
    ("What does the attached document say about CEO succession?",        "attachment"),
    ("What is on page 2 of the PDF?",                                   "attachment"),
    ("Summarise the attached proposal",                                  "attachment"),
    ("What are the terms in the attached contract?",                     "attachment"),
    ("What does the confidential draft say?",                            "attachment"),
    ("What is in the attached succession plan document?",                "attachment"),
    ("Can you read the attachment for me?",                              "attachment"),
    ("What does the board document cover on page 1?",                    "attachment"),
    ("Describe the contents of the file attached to this email",         "attachment"),
    ("What transition timeline is outlined in the attached file?",       "attachment"),
    ("Is there a page 3 in the document?",                              "attachment"),
    ("What does the report say about Jeffrey Skilling?",                 "attachment"),
    ("What are the key figures in the attached spreadsheet?",            "attachment"),
    ("What is the first page of the attachment about?",                  "attachment"),
    ("Read the attached memo for me",                                    "attachment"),
    ("What does the enclosed document recommend?",                       "attachment"),

    # ── status ────────────────────────────────────────────────────────
    # Status = specific outcome, current state, resolution of ONE issue
    ("What is the current status of the succession plan?",               "status"),
    ("Has this been resolved or is it still pending?",                   "status"),
    ("What happened with the executive succession planning?",            "status"),
    ("What is the latest update on the CEO transition?",                 "status"),
    ("Is the transition complete?",                                      "status"),
    ("What was decided in the end?",                                     "status"),
    ("What is the overall outcome of this discussion?",                  "status"),
    ("What is the key takeaway from this thread?",                       "status"),
    ("What did they conclude about the FERC price cap?",                 "status"),
    ("Was this issue ever resolved?",                                    "status"),
    ("What is the current state of the negotiations?",                   "status"),
    ("What is the final decision?",                                      "status"),
    ("What action was taken after this email?",                          "status"),
    ("What is the result of this discussion?",                           "status"),
    ("Is this matter still open?",                                       "status"),
    ("What was the conclusion reached?",                                 "status"),

    # ── summary ───────────────────────────────────────────────────────
    # Summary = comprehensive overview of the WHOLE thread / conversation
    ("Summarise the whole email thread",                                 "summary"),
    ("Give me an overview of this conversation",                         "summary"),
    ("What happened in this thread?",                                    "summary"),
    ("Recap the main points of this discussion",                         "summary"),
    ("Describe the entire thread",                                       "summary"),
    ("What is this thread about?",                                       "summary"),
    ("What were the key points discussed?",                              "summary"),
    ("Summarise this thread",                                            "summary"),
    ("Give me a summary of what happened",                               "summary"),
    ("What is this email chain about?",                                  "summary"),
    ("Brief me on this thread",                                          "summary"),
    ("Tell me what this whole conversation is about",                    "summary"),
    ("Give me an overview of everything discussed here",                 "summary"),
    ("What does this entire email chain cover?",                         "summary"),
    ("What is the gist of this thread?",                                 "summary"),
    ("Walk me through what happened in this email chain",                "summary"),
]

LABELS = ["approval", "timeline", "person", "attachment", "status", "summary"]


# ──────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────

_classifier = None


def get_intent_classifier():
    """Return the trained SetFit classifier (loaded once, cached in-process)."""
    global _classifier
    if _classifier is None:
        _classifier = _load_or_fallback()
    return _classifier


def predict_label(query: str) -> str:
    """Predict the intent cluster label for a query. Safe to call from agents.py."""
    clf = get_intent_classifier()
    return clf.predict_label(query)


# ──────────────────────────────────────────────────────────────────────
# Classifier wrappers
# ──────────────────────────────────────────────────────────────────────

class _SetFitWrapper:
    def __init__(self, model):
        self._model = model

    def predict_label(self, query: str) -> str:
        result = self._model.predict([query])
        label = result[0] if hasattr(result, "__len__") else str(result)
        return str(label)


class _CentroidFallback:
    """Fallback to centroid averaging when SetFit model is not available."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._model = SentenceTransformer(BASE_MODEL)
        self._np = np
        centroids: dict[str, list] = {}
        for text, label in TRAIN_EXAMPLES:
            centroids.setdefault(label, []).append(text)
        self._centroids = {
            label: self._model.encode(texts, normalize_embeddings=True).mean(axis=0)
            for label, texts in centroids.items()
        }

    def predict_label(self, query: str) -> str:
        import numpy as np
        q = self._model.encode(query, normalize_embeddings=True)
        scores = {label: float(np.dot(q, c)) for label, c in self._centroids.items()}
        return max(scores, key=scores.get)


def _load_or_fallback():
    if MODEL_SAVE_DIR.exists():
        try:
            from setfit import SetFitModel
            model = SetFitModel.from_pretrained(str(MODEL_SAVE_DIR))
            print(f"[intent] SetFit model loaded from {MODEL_SAVE_DIR}")
            return _SetFitWrapper(model)
        except Exception as e:
            print(f"[intent] SetFit load failed ({e}), using centroid fallback")
    else:
        print(f"[intent] No trained model at {MODEL_SAVE_DIR} — using centroid fallback. "
              f"Run `python utils/intent_classifier.py` to train.")
    return _CentroidFallback()


# ──────────────────────────────────────────────────────────────────────
# Training script — run once: python utils/intent_classifier.py
# ──────────────────────────────────────────────────────────────────────

def train_and_save():
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    texts  = [t for t, _ in TRAIN_EXAMPLES]
    labels = [l for _, l in TRAIN_EXAMPLES]

    label2id = {l: i for i, l in enumerate(LABELS)}

    dataset = Dataset.from_dict({
        "text":  texts,
        "label": [label2id[l] for l in labels],
    })

    model = SetFitModel.from_pretrained(
        BASE_MODEL,
        labels=LABELS,
    )

    args = TrainingArguments(
        num_epochs=3,
        batch_size=16,
        num_iterations=25,
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    print(f"Training SetFit on {len(TRAIN_EXAMPLES)} examples, {len(LABELS)} classes...")
    trainer.train()

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_SAVE_DIR))
    print(f"Model saved to {MODEL_SAVE_DIR}")

    # Comprehensive sanity check covering all hard cases
    test_queries = [
        ("Who approved the deal?",                              "approval"),
        ("When was this sent?",                                "timeline"),
        ("Who wrote this email?",                              "person"),
        ("Who are the people involved in this thread?",        "person"),
        ("Who are the key people in this thread?",             "person"),
        ("What is in the attachment?",                         "attachment"),
        ("What is the current status of this?",                "status"),
        ("Was this resolved?",                                 "status"),
        ("Summarise the whole thread",                         "summary"),
        ("Summarise this thread",                              "summary"),
        ("Give me an overview of this conversation",           "summary"),
        ("What is this thread about?",                         "summary"),
        ("What happened in this thread?",                      "summary"),
        ("What were the key points discussed?",                "summary"),
        ("Brief me on this thread",                            "summary"),
    ]
    print("\nSanity check:")
    wrapper = _SetFitWrapper(model)
    correct = 0
    for q, expected in test_queries:
        pred = wrapper.predict_label(q)
        ok = pred == expected
        if ok:
            correct += 1
        status = "OK" if ok else f"WRONG (expected {expected})"
        print(f"  [{status}] '{q}' -> {pred}")
    print(f"\nAccuracy: {correct}/{len(test_queries)} = {correct/len(test_queries):.0%}")


if __name__ == "__main__":
    train_and_save()
