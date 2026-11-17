"""Module suy luận (Inference) của AutoClaim-AI: TwoStagePipeline & ChangeDetection."""

from ai.inference.pipeline import (
    DamageResult,
    MappedDetection,
    PartResult,
    PipelineResponse,
    TwoStagePipeline,
    draw_annotations,
)

__all__ = [
    "DamageResult",
    "MappedDetection",
    "PartResult",
    "PipelineResponse",
    "TwoStagePipeline",
    "draw_annotations",
]
