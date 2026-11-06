"""EcoGate-Audio models package."""

from src.models.audio_models import (
    LightweightStudentCNN,
    MFCCBaselineModel,
    ModelOutput,
    StrongExpertAudioModel,
    sigmoid,
    softmax,
)

try:
    from src.models.pytorch_models import (
        ExpertTransformer,
        StudentCNN,
        seed_everything,
        train_model,
    )
    _PYTORCH_MODELS_AVAILABLE = True
except ImportError:
    _PYTORCH_MODELS_AVAILABLE = False

__all__ = [
    "LightweightStudentCNN",
    "MFCCBaselineModel",
    "ModelOutput",
    "StrongExpertAudioModel",
    "sigmoid",
    "softmax",
    # PyTorch models (available when torch is installed)
    "StudentCNN",
    "ExpertTransformer",
    "seed_everything",
    "train_model",
]
