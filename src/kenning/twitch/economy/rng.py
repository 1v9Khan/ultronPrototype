"""Provably-fair commit-reveal RNG (HMAC-SHA256), server-decided outcomes.

Every game outcome is decided SERVER-SIDE *before* any overlay animation, and is
independently verifiable after the fact (the ``!verify`` command). The scheme
(MASTER.md SLICE 9 / S_report):

  1. ``new_round()`` mints a cryptographically-random ``server_seed``
     (``secrets.token_bytes``) and publishes ``commit = sha256(server_seed)``
     BEFORE the round. The seed itself is withheld until reveal — so the house
     cannot change the outcome after seeing the bet, and the viewer cannot
     predict it.
  2. The viewer (or a default) supplies a ``client_seed`` and a per-bet ``nonce``
     so the streamer can't grind seeds either; the outcome is a deterministic
     function of (server_seed, client_seed, nonce).
  3. ``outcome(...)`` derives an unbiased integer in ``[0, n)`` from
     ``HMAC-SHA256(server_seed, f"{client_seed}:{nonce}")`` using rejection
     sampling (no modulo bias). ``weighted_choice(...)`` maps the same uniform
     draw onto a weighted segment list — this is where the wheel's winning index
     / target angle is decided, server-side, before the overlay spins.
  4. After the round, ``server_seed`` is revealed; ``verify(commit, server_seed)``
     re-checks the commitment and anyone can re-run ``outcome`` to reproduce
     every result.

ANTICHEAT (BR-P1): stdlib only — ``hashlib`` / ``hmac`` / ``secrets``. The seed
source is ``secrets`` (os CSPRNG); no ``random`` module (not cryptographically
sound) anywhere on the money path.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger("kenning.twitch.economy.rng")

__all__ = ["ProvablyFairRNG", "RoundCommit", "RngError"]

# 64 hex chars of HMAC-SHA256 -> 256 bits of uniform entropy per (seed,msg).
_DIGEST_BITS = 256


class RngError(Exception):
    """Invalid RNG parameters (bad seed, n<=0, empty/negative weights)."""


@dataclass(frozen=True)
class RoundCommit:
    """The pre-round commitment a viewer can record before betting."""

    server_seed: str   # hex; SECRET until reveal
    commit: str        # sha256(server_seed) as hex; published BEFORE the round


def _validate_hex_seed(server_seed: str, field: str = "server_seed") -> str:
    if not isinstance(server_seed, str):
        raise RngError(f"{field} must be a hex str, got {type(server_seed).__name__}")
    s = server_seed.strip().lower()
    if not s:
        raise RngError(f"{field} must be non-empty")
    try:
        bytes.fromhex(s)
    except ValueError as e:
        raise RngError(f"{field} is not valid hex: {e}") from e
    if len(s) % 2 != 0:
        raise RngError(f"{field} has an odd hex length")
    return s


def _validate_seed_str(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise RngError(f"{field} must be a str, got {type(value).__name__}")
    if len(value) > 4096:
        raise RngError(f"{field} too long ({len(value)} chars)")
    return value


def sha256_hex(data: str | bytes) -> str:
    """``sha256`` hex digest of a str (utf-8) or bytes — the commit primitive."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, (bytes, bytearray)):
        raise RngError(f"sha256 input must be str/bytes, got {type(data).__name__}")
    return hashlib.sha256(data).hexdigest()


def _hmac_digest(server_seed: str, message: str) -> bytes:
    """``HMAC-SHA256(server_seed_bytes, message_bytes)`` — the per-draw entropy.

    The key is the raw seed BYTES (decoded from hex), the message is the utf-8
    of ``f"{client_seed}:{nonce}"``. This is the single source of randomness for
    every outcome so ``!verify`` is a one-liner re-derivation.
    """
    key = bytes.fromhex(server_seed)
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _message(client_seed: str, nonce: int) -> str:
    if isinstance(nonce, bool) or not isinstance(nonce, int):
        raise RngError(f"nonce must be a non-bool int, got {type(nonce).__name__}")
    if nonce < 0:
        raise RngError(f"nonce must be >= 0, got {nonce}")
    cs = _validate_seed_str(client_seed, "client_seed")
    return f"{cs}:{nonce}"


class ProvablyFairRNG:
    """Commit-reveal HMAC-SHA256 RNG. Stateless except for an optional default
    ``client_seed`` (used when a caller does not supply one)."""

    def __init__(self, *, default_client_seed: str = "ultron") -> None:
        self._default_client_seed = _validate_seed_str(
            default_client_seed, "default_client_seed"
        )

    # -- commit / reveal ------------------------------------------------------
    def new_round(self, *, seed_bytes: int = 32) -> RoundCommit:
        """Mint a fresh secret ``server_seed`` and its public ``commit``.

        :param seed_bytes: entropy of the server seed (>=16). 32 = 256 bits.
        """
        if isinstance(seed_bytes, bool) or not isinstance(seed_bytes, int):
            raise RngError("seed_bytes must be an int")
        if seed_bytes < 16:
            raise RngError("seed_bytes must be >= 16 for a secure commitment")
        server_seed = secrets.token_bytes(seed_bytes).hex()
        commit = sha256_hex(server_seed)
        logger.info("rng new_round commit=%s (seed withheld until reveal)", commit)
        return RoundCommit(server_seed=server_seed, commit=commit)

    def commit_for(self, server_seed: str) -> str:
        """``sha256(server_seed)`` — recompute a commitment from a known seed."""
        s = _validate_hex_seed(server_seed)
        return sha256_hex(s)

    def verify(self, commit: str, server_seed: str) -> bool:
        """True iff ``commit == sha256(server_seed)`` (constant-time compare).

        The viewer-facing ``!verify`` check: the revealed seed must hash to the
        commitment that was published before the round. Never raises on a
        malformed input — a bad seed/commit is simply not a match (False)."""
        try:
            s = _validate_hex_seed(server_seed)
        except RngError:
            return False
        if not isinstance(commit, str) or not commit.strip():
            return False
        recomputed = sha256_hex(s)
        ok = hmac.compare_digest(recomputed, commit.strip().lower())
        if not ok:
            logger.info("rng verify FAILED commit=%s", commit[:16])
        return ok

    # -- outcomes -------------------------------------------------------------
    def outcome(
        self, server_seed: str, client_seed: str, nonce: int, n: int
    ) -> int:
        """An unbiased integer in ``[0, n)``, deterministic for fixed inputs.

        Uses rejection sampling over the 256-bit HMAC digest so there is NO
        modulo bias (the failure mode of ``digest % n``). For the tiny ``n`` of
        real games (2-50) the rejection probability is negligible, but extra
        entropy is derived by appending a counter so the function always
        terminates and stays deterministic.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise RngError(f"n must be a non-bool int, got {type(n).__name__}")
        if n <= 0:
            raise RngError(f"n must be >= 1, got {n}")
        if n == 1:
            return 0
        seed = _validate_hex_seed(server_seed)
        base_msg = _message(client_seed, nonce)

        # Largest multiple of n that fits in _DIGEST_BITS — draws above it are
        # rejected to keep the distribution exactly uniform.
        limit = 1 << _DIGEST_BITS
        max_unbiased = limit - (limit % n)

        counter = 0
        while True:
            msg = base_msg if counter == 0 else f"{base_msg}:{counter}"
            digest = _hmac_digest(seed, msg)
            value = int.from_bytes(digest, "big")
            if value < max_unbiased:
                return value % n
            counter += 1
            if counter > 10_000:  # pragma: no cover - astronomically unreachable
                # Defensive: never spin forever. Fall back to the modulo (the
                # residual bias here is < 2**-240, far below any practical limit).
                logger.warning("rng outcome rejection sampling hit cap; using modulo")
                return value % n

    def uniform_unit(
        self, server_seed: str, client_seed: str, nonce: int
    ) -> float:
        """A deterministic float in ``[0, 1)`` from the same HMAC draw.

        Used for continuous mappings (e.g. an exact target angle within an arc).
        """
        seed = _validate_hex_seed(server_seed)
        msg = _message(client_seed, nonce)
        digest = _hmac_digest(seed, msg)
        value = int.from_bytes(digest, "big")
        return value / float(1 << _DIGEST_BITS)

    def weighted_choice(
        self,
        server_seed: str,
        client_seed: str,
        nonce: int,
        weights: Sequence[float],
    ) -> int:
        """Index into ``weights`` chosen with probability proportional to weight.

        This is where the spin-the-wheel / slot winner is DECIDED, server-side,
        before any animation. Deterministic for fixed inputs; verifiable via
        ``!verify`` by re-running with the revealed seed.

        Weights must be finite and >= 0 with a positive sum. The uniform draw
        ``u in [0,1)`` is scaled by the total weight and the cumulative band it
        lands in selects the index (no float-equality edge: the last band
        absorbs any rounding via a strict-less cascade with a final fallthrough).
        """
        if weights is None or len(weights) == 0:
            raise RngError("weights must be a non-empty sequence")
        norm: list[float] = []
        total = 0.0
        for i, w in enumerate(weights):
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise RngError(f"weight[{i}] must be a number, got {type(w).__name__}")
            wf = float(w)
            if wf != wf or wf in (float("inf"), float("-inf")):
                raise RngError(f"weight[{i}] must be finite, got {w!r}")
            if wf < 0:
                raise RngError(f"weight[{i}] must be >= 0, got {wf}")
            norm.append(wf)
            total += wf
        if total <= 0.0:
            raise RngError("sum of weights must be > 0")

        u = self.uniform_unit(server_seed, client_seed, nonce)
        target = u * total
        cumulative = 0.0
        for i, wf in enumerate(norm):
            cumulative += wf
            if target < cumulative:
                return i
        # Float rounding fallthrough: return the last NON-ZERO-weight index.
        for i in range(len(norm) - 1, -1, -1):
            if norm[i] > 0.0:
                return i
        raise RngError("no positive-weight index (unreachable: total>0 checked)")

    @property
    def default_client_seed(self) -> str:
        return self._default_client_seed
