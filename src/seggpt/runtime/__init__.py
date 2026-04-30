"""Layer 1: SegGPT runtime kernel.

Ported from generative-services-server. Stateful three-step API
(target / prompt / reset). Internal use only; downstream consumers
should import from ``seggpt.api`` (Layer 2) instead.
"""
