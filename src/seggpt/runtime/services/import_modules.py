"""Lazy imports for heavy third-party modules referenced by the kernel.

Slimmed port of ``generative_services.services.import_modules``. Only
the lazy handles touched by ``seggpt_service.py`` and
``seggpt_model.py`` are kept; upstream additionally lazy-loaded
``groundingdino``, ``segment_anything``, and ``supervision`` for other
services that this repo does not port.
"""
from typing import TYPE_CHECKING

from seggpt.runtime.utils.lazy_import import LazyModuleImporter

if TYPE_CHECKING:
    import detectron2.layers
    import fairscale.nn.checkpoint
    import fvcore.nn.weight_init
    import timm.layers
    import timm.models
    import torch
    import torchvision
    import torchvision.transforms.functional

    import_torch = torch
    import_torchvision = torchvision
    import_torchvision_transforms_functional = torchvision.transforms.functional
    import_detectron2_layers = detectron2.layers
    import_fvcore_nn_weight_init = fvcore.nn.weight_init
    import_fairscale_nn_checkpoint = fairscale.nn.checkpoint
    import_timm_layers = timm.layers
    import_timm_models = timm.models
else:
    import_torch = LazyModuleImporter("torch")
    import_torchvision = LazyModuleImporter("torchvision")
    import_torchvision_transforms_functional = LazyModuleImporter("torchvision.transforms.functional")
    import_detectron2_layers = LazyModuleImporter("detectron2.layers")
    import_fvcore_nn_weight_init = LazyModuleImporter("fvcore.nn.weight_init")
    import_fairscale_nn_checkpoint = LazyModuleImporter("fairscale.nn.checkpoint")
    import_timm_layers = LazyModuleImporter("timm.layers")
    import_timm_models = LazyModuleImporter("timm.models")
