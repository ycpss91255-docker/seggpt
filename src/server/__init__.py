"""Layer 3: FastAPI HTTP wrapper around Layer 2.

Provides a thin HTTP surface for external clients and interactive
testing. Returns the same raw output as ``seggpt.api.SegGPTBackend``,
just serialised over HTTP.
"""
