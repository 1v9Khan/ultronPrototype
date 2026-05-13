"""Rule implementations for the runtime tool-call validator.

Each module under :mod:`ultron.safety.rules` corresponds to one
category from the user's 2026-05-12 restriction list:

* ``base`` -- abstract :class:`Rule` base class.
* ``category_k`` -- Category K (Ultron self-protection / meta).
  Phase 2.

Phases 3-5 add:

* ``category_a`` -- filesystem destruction
* ``category_b`` -- privilege escalation + system config
* ``category_c`` -- security perimeter
* ``category_d`` -- credential / secret access
* ``category_e`` -- system stability
* ``category_f`` -- repository / data integrity
* ``category_g`` -- resource exhaustion
* ``category_h`` -- untrusted code execution
* ``category_i`` -- outbound impact
* ``category_j`` -- data exfiltration
* ``category_m`` -- persistence mechanisms
* ``category_n`` -- process / memory manipulation
* ``category_o`` -- anti-forensics
* ``category_p`` -- AV / EDR tampering
* ``category_q`` -- containers + virtualization
* ``category_r`` -- sensors + input
* ``category_s`` -- AI-specific tampering
* ``cap_carveouts`` -- Cap-1 .. Cap-4 capability allowances

Each ``category_X`` module exports a ``build_category_X_rules() -> list[Rule]``
factory that the validator-builder calls during construction.
"""

from __future__ import annotations

from ultron.safety.rules.base import Rule

__all__ = ["Rule"]
