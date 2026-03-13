"""Framework modular para pipeline jurídica (etapas 1-4)."""

from .stage1_document_type import run_stage1
from .stage2_preprocessing import run_stage2
from .stage3_finetune import run_stage3_finetune
from .stage3_embeddings import run_stage3_embeddings
from .stage4_ner import run_stage4_ner

__all__ = [
    "run_stage1",
    "run_stage2",
    "run_stage3_finetune",
    "run_stage3_embeddings",
    "run_stage4_ner",
]
