"""Layer 2: stable Python API for SegGPT inference.

Stateless one-shot wrapper around Layer 1. The entry point downstream
consumers (Phase 0 benchmark harness, ROS 2 backends, etc.) should
import. Returns raw SegGPT output without any quality wrapping.
"""
