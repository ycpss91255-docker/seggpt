"""Service-layer helpers used by the abstract base + SegGPT kernel.

Direct port of ``generative_services.services.utils``: 3 functions,
no slimming necessary.
"""
import inspect

from seggpt.runtime.services.import_modules import import_torch as torch
from seggpt.runtime.utils.environment_variables import SEGGPT_PRECISION, USE_CUDA

# Accepted SEGGPT_PRECISION values (#14). ``fp16`` (autocast) is tracked in #12
# and deliberately rejected until it lands, so a typo never silently runs fp32.
PRECISION_MODES = ("fp32", "tf32")


def torch_use_cuda() -> str:
    """Return ``"cuda"`` if USE_CUDA env is on and a device is visible, else ``"cpu"``."""
    return "cuda" if USE_CUDA.get() and torch.cuda.is_available() else "cpu"


def configure_precision(mode: str) -> str:
    """Apply a SegGPT numeric-precision mode to the torch backends and return it normalised.

    ``fp32``: PyTorch defaults — matmul TF32 off (cuDNN conv TF32 stays at
    torch's default). ``tf32``: TensorFloat-32 for matmul + cuDNN, i.e. the
    Tensor Cores do the ViT-L GEMMs with fp32 tensors in and out (1.75x on
    Jetson AGX Orin, mask agreement with fp32 0.9994 over 35 images).
    A no-op on CPU. Raises ``ValueError`` for anything else.
    """
    normalised = str(mode).strip().lower()
    if normalised not in PRECISION_MODES:
        raise ValueError(
            f"SEGGPT_PRECISION must be one of {list(PRECISION_MODES)}, got {mode!r}"
            + (" (fp16 autocast is not supported yet, see #12)" if normalised == "fp16" else "")
        )
    tf32 = normalised == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if tf32:
        torch.backends.cudnn.allow_tf32 = True
    return normalised


def apply_precision_from_env() -> str:
    """Read ``SEGGPT_PRECISION`` and apply it; returns the normalised mode."""
    return configure_precision(SEGGPT_PRECISION.get())


def contains_var_keyword(signature: inspect.Signature) -> bool:
    """Return whether ``signature`` declares ``**kwargs``."""
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def get_var_keyword(signature: inspect.Signature) -> str:
    """Return the name of the ``**kwargs`` parameter, or ``""`` if none.

    Raises ``ValueError`` if the signature declares more than one
    variadic-keyword parameter (Python disallows this so it should never
    fire in practice; kept as defensive parity with upstream).
    """
    vars_ = [
        p.name
        for p in signature.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    if len(vars_) == 0:
        return ""
    if len(vars_) > 1:
        raise ValueError("The signature contains multiple variable keyword arguments.")
    return vars_[0]
