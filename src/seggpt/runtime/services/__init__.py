"""Layer 1 service kernel.

Stateful three-step API (``target`` -> ``prompt`` -> ``reset``).
Internal use only; downstream consumers should import from
``seggpt.api`` (Layer 2) instead of reaching directly into this
package.
"""
