"""
probe_utils.py

Builds the answerable/unanswerable probe pair from HellaSwag (see the two
probe_hellaswag_*.yaml files next to this file).

Purpose: entropy-style signals cannot distinguish "this question is ambiguous"
from "the model cannot know the answer". The epistemic term claims to measure
the latter. This probe tests that claim directly: both variants share the same
contexts, but in the unanswerable variant the TRUE ending is replaced by the
true ending of ANOTHER document, so no offered choice is correct and the
question is unanswerable in principle. A useful lack-of-knowledge signal
should separate the two variants; probe.py measures how well each signal does.

The swap is deterministic (doc i receives the gold ending of doc i+1,
cyclically), so runs are reproducible without any RNG. The stored label of the
unanswerable variant still points at the swapped slot; accuracy on that task
is meaningless by design (there is no correct answer). The analysis only uses
the per-choice log-likelihoods.
"""

from lm_eval.tasks.hellaswag.utils import process_docs as hellaswag_process_docs


def process_docs_answerable(dataset):
    """The unchanged HellaSwag preprocessing - the answerable twin."""
    return hellaswag_process_docs(dataset)


def process_docs_unanswerable(dataset):
    """HellaSwag with the true ending swapped out: no offered choice is correct."""
    docs = hellaswag_process_docs(dataset)
    golds = [doc["choices"][doc["gold"]] for doc in docs]

    def _swap(doc, idx):
        choices = list(doc["choices"])
        choices[doc["gold"]] = golds[(idx + 1) % len(golds)]
        return {"query": doc["query"], "choices": choices, "gold": doc["gold"]}

    return docs.map(_swap, with_indices=True)
