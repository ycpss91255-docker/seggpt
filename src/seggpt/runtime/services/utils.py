"""Service-layer helpers used by the abstract base + SegGPT kernel.

Direct port of ``generative_services.services.utils``: 3 functions,
no slimming necessary.
"""
import inspect

from seggpt.runtime.services.import_modules import import_torch as torch
from seggpt.runtime.utils.environment_variables import USE_CUDA


def torch_use_cuda() -> str:
    """Return ``"cuda"`` if USE_CUDA env is on and a device is visible, else ``"cpu"``."""
    return "cuda" if USE_CUDA.get() and torch.cuda.is_available() else "cpu"


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
