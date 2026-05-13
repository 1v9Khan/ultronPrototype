"""Category J -- Data exfiltration (OUT-gate).

J1 -- upload local files to external services without explicit intent.
J2 -- post file contents to chat / pastebins / gists.
J3 -- send audit logs / logs/*.jsonl off-machine.
J4 -- DNS-based exfil patterns.
J5 -- ICMP exfil patterns.
J6 -- cloud-storage SDK uploads to non-approved destinations.
J7 -- clipboard writes that look like crypto-address substitution.
J8 -- screen-context outflow (NEEDS_EXPLICIT_INTENT plus path matcher).
J9 -- email/IM image attachments sourced from screen-capture cache.
"""

from __future__ import annotations

import re

from ultron.safety.rules.base import CommandPatternRule, Rule
from ultron.safety.validator import RuleResult, Verdict


def build_category_j_rules() -> list[Rule]:
    """Factory for Category J rules."""
    rules: list[Rule] = []

    # J2: posting file contents to chat / pastebins / gists.
    rules.append(
        CommandPatternRule(
            rule_id="J2",
            description="post file contents to pastebins / gists / chat",
            category="J",
            patterns=[
                r"\bpastebin\.com/api/api_post\.php\b",
                r"\bapi\.github\.com/gists\b.*POST",
                r"\bgist\.github\.com\b.*POST",
                r"\btransfer\.sh\b",
                r"\b0x0\.st\b",
                r"\bbpa\.st\b",
                r"\bdiscord\.com/api/.*/messages\b.*--data-binary",
                r"\bslack\.com/api/files\.upload\b",
            ],
        )
    )

    # J3: sending audit logs off-machine.
    # Pattern combines a log path with an upload verb.
    rules.append(
        CommandPatternRule(
            rule_id="J3",
            description="send audit logs / logs/*.jsonl off-machine",
            category="J",
            patterns=[
                # rclone / aws / gcloud uploading from logs/
                r"\b(rclone|aws|gcloud|az)\s+.*\blogs/[\w./-]*\.jsonl\b",
                r"\bcurl\s+.*--data-binary\s+@logs/.*\.jsonl\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        )
    )

    # J4: DNS-based exfil.
    rules.append(
        CommandPatternRule(
            rule_id="J4",
            description="DNS-based exfiltration patterns",
            category="J",
            patterns=[
                # Very long encoded subdomain label
                r"\b[a-z0-9+/=]{50,}\.[a-z0-9-]+\.[a-z]{2,}\b",
                # nslookup with very long TXT queries
                r"\bnslookup\s+-type=txt\s+[a-z0-9]{50,}",
                # dig with TXT to unknown domains
                r"\bdig\s+TXT\s+[a-z0-9-]{50,}",
            ],
        )
    )

    # J5: ICMP exfil.
    rules.append(
        CommandPatternRule(
            rule_id="J5",
            description="ICMP exfiltration patterns",
            category="J",
            patterns=[
                r"\bping\s+-l\s+\d{4,}\b",          # large ICMP payloads
                r"\bping\s+-s\s+\d{4,}\b",
                r"\bSystem\.Net\.NetworkInformation\.Ping\b.*\.Send\b.*new\s+byte\[\d{3,}\]",
            ],
        )
    )

    # J6: cloud-storage uploads to non-approved destinations.
    rules.append(
        CommandPatternRule(
            rule_id="J6",
            description="cloud-storage uploads to non-approved destinations",
            category="J",
            patterns=[
                r"\baws\s+s3\s+cp\b",
                r"\baws\s+s3\s+sync\b",
                r"\bgcloud\s+storage\s+(cp|rsync)\b",
                r"\baz\s+storage\s+blob\s+upload\b",
                r"\brclone\s+(copy|copyto|sync)\s+\S+\s+\S+:",   # local -> remote
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        )
    )

    # J7: clipboard writes that look like crypto-address substitution.
    # The threat: clipper malware replaces a Bitcoin address copied by
    # the user with the attacker's. Detected when the new clipboard
    # content matches a crypto-address format.
    class _J7ClipperPatternRule(Rule):
        rule_id = "J7"
        description = "clipboard writes that look like crypto-address substitution"
        category = "J"

        _ADDRESS_PATTERNS = [
            # BTC legacy + bech32
            r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$",
            r"^bc1[a-z0-9]{6,87}$",
            # ETH
            r"^0x[a-fA-F0-9]{40}$",
            # Solana
            r"^[1-9A-HJ-NP-Za-km-z]{32,44}$",
            # Litecoin
            r"^(L|M|ltc1)[a-zA-Z0-9]{25,40}$",
            # IBAN
            r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$",
        ]
        _CLIPBOARD_WRITE_PATTERN = re.compile(
            r"\b(Set-Clipboard|pbcopy|xclip|xsel|pyperclip\.copy|clipboard\.SetText)\b",
            re.IGNORECASE,
        )

        def evaluate(self, ctx, *, policy, resolver):  # noqa: ARG002
            haystack = ctx.tool_name + "\n" + "\n".join(
                str(v) for v in ctx.arguments.values() if v is not None
            )
            if not self._CLIPBOARD_WRITE_PATTERN.search(haystack):
                return RuleResult(
                    rule_id=self.rule_id,
                    verdict=Verdict.ALLOW,
                    reason="not a clipboard-write call",
                )
            # If the new clipboard content matches a crypto-address
            # format, that's a high-precision red flag for clipper
            # malware behaviour.
            for v in ctx.arguments.values():
                if not isinstance(v, str):
                    continue
                for addr_pat in self._ADDRESS_PATTERNS:
                    if re.match(addr_pat, v.strip()):
                        return RuleResult(
                            rule_id=self.rule_id,
                            verdict=Verdict.BLOCK_HARD,
                            reason=(
                                "clipboard write of a crypto-address / IBAN "
                                "(clipper-malware pattern)"
                            ),
                            context={
                                "pattern": addr_pat,
                                "value_preview": v.strip()[:80],
                            },
                        )
            return RuleResult(
                rule_id=self.rule_id,
                verdict=Verdict.ALLOW,
                reason="clipboard write content didn't match address shape",
            )

    rules.append(_J7ClipperPatternRule())

    return rules
