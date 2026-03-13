"""Framework modular para pipeline jurídica (stage1_1, stage1_2, stage2, stage5)."""

from .stage1_1_document_type import run_stage1_1
from .stage1_2_preprocessing import run_stage1_2
from .stage2_embeddings import run_stage2_embeddings
from .stage2_finetune import run_stage2_finetune
from .stage5_ner import run_stage5_ner

__all__ = [
    "run_stage1_1",
    "run_stage1_2",
    "run_stage2_embeddings",
    "run_stage2_finetune",
    "run_stage5_ner",
]
