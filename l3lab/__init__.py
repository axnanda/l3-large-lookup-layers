"""l3lab: reference L3 layer, allocation, tokenizers, spiking primitives, models (+ data/train/WW analysis)."""
from . import allocation, l3_ref, models, snn, tokenizers  # noqa: F401
from .allocation import allocate_by_frequency, train_lzw_generic  # noqa: F401
from .l3_ref import AllocTable, L3Ref  # noqa: F401
from .models import L3CharLM, L3Classifier, MLP3, SNNL3Classifier, SNNMLP3  # noqa: F401
from .snn import LIF, spike, surrogate_sigmoid  # noqa: F401
from .tokenizers import CharTokenizer, PatchVQTokenizer  # noqa: F401

# Modules written by the other build agent; tolerate their absence so the core package always imports.
try:  # pragma: no cover
    from . import data  # noqa: F401
except ImportError:
    pass
try:  # pragma: no cover
    from . import train  # noqa: F401
except ImportError:
    pass
try:  # pragma: no cover
    from . import ww_analysis  # noqa: F401
except ImportError:
    pass
