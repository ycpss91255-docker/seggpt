"""SegGPT visual prompt segmentation backend.

Three-layer architecture:

* ``seggpt.runtime`` — Layer 1 kernel (stateful target/prompt/reset).
  Internal use only; downstream consumers should not import this.
* ``seggpt.api`` — Layer 2 stable Python API. The supported entry
  point for downstream consumers (Phase 0 benchmark harness, ROS 2
  backends, etc.). Returns raw SegGPT output.
* ``seggpt.server`` — Layer 3 FastAPI HTTP wrapper around Layer 2,
  for external clients and interactive testing.
"""

__version__ = "0.0.0"
