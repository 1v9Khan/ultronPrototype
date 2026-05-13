"""Category Q -- Containers and virtualization.

Q1 -- privileged docker run / host-namespace mounts / capability adds.
Q2 -- Docker socket access from non-Docker-Desktop process.
Q3 -- WSL config changes affecting all distros.
Q4 -- Hyper-V VM create/modify with passthrough hardware.
Q5 -- standard ``docker run`` for project images -- LOG_ONLY.
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule
from ultron.safety.validator import Verdict


def build_category_q_rules() -> list[Rule]:
    """Factory for Category Q rules."""
    return [
        # Q1: privileged docker run shapes.
        CommandPatternRule(
            rule_id="Q1",
            description="privileged docker run / host-namespace / sensitive mounts",
            category="Q",
            patterns=[
                r"\bdocker\s+run\b.*--privileged\b",
                r"\bdocker\s+run\b.*--pid=host\b",
                r"\bdocker\s+run\b.*--net=host\b",
                r"\bdocker\s+run\b.*--userns=host\b",
                r"\bdocker\s+run\b.*--cap-add=(SYS_ADMIN|ALL|NET_ADMIN|SYS_PTRACE)\b",
                # Mounting sensitive host paths
                r"\bdocker\s+run\b.*-v\s+/:/[^\s]*",         # root /
                r"\bdocker\s+run\b.*-v\s+c:\\\\:",            # Windows C:
                r"\bdocker\s+run\b.*-v\s+(/var/run/docker\.sock|//\.//pipe/docker_engine):",
                r"\bdocker\s+run\b.*-v\s+/proc[:/]",
                r"\bdocker\s+run\b.*-v\s+/sys[:/]",
                r"\bdocker\s+run\b.*-v\s+/etc[:/]",
            ],
        ),
        # Q2: Docker socket access.
        CommandPatternRule(
            rule_id="Q2",
            description="Docker socket access from non-Docker-Desktop process",
            category="Q",
            patterns=[
                # Reading / writing the Unix socket
                r"\\?\b/var/run/docker\.sock\b",
                # Windows named pipe
                r"\\\\\\\\\.\\\\pipe\\\\docker_engine",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # Q3: WSL config changes.
        CommandPatternRule(
            rule_id="Q3",
            description="WSL config changes / image export/import",
            category="Q",
            patterns=[
                r"\bwsl\s+--import\b",
                r"\bwsl\s+--export\b",
                r"\bwsl\s+--unregister\b",
                r"\bwsl\s+--shutdown\b",
                # .wslconfig at user-level affects all distros
                r"%userprofile%\\\\\.wslconfig",
                r"\$home\\\\\.wslconfig",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # Q4: Hyper-V VM operations.
        CommandPatternRule(
            rule_id="Q4",
            description="Hyper-V VM creation / modification with passthrough",
            category="Q",
            patterns=[
                r"\bNew-VM\b",
                r"\bSet-VM\b.*-PassThru\b",
                r"\bAdd-VMHostAssignableDevice\b",
                r"\bDismount-VMHostAssignableDevice\b",
                r"\bSet-VM\b.*-ExposeVirtualizationExtensions\s+\$true\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
    ]
