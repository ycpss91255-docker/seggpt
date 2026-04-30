"""Lazy imports for sibling modules inside ``seggpt.runtime.services``.

Breaks the import cycle between ``seggpt_service`` and ``seggpt_model``:
``seggpt_service`` only touches the model lazily at first inference, so
deferring the import keeps ``seggpt.runtime`` import-time cheap. Upstream
additionally lazy-loaded ``fixed_color_palette`` and
``automatic_mask_generator`` for other services that this repo does
not port.
"""
from typing import TYPE_CHECKING

from seggpt.runtime.utils.lazy_import import LazyModuleImporter

if TYPE_CHECKING:
    import seggpt.runtime.services.seggpt_model

    import_seggpt_model = seggpt.runtime.services.seggpt_model
else:
    import_seggpt_model = LazyModuleImporter("seggpt.runtime.services.seggpt_model")
