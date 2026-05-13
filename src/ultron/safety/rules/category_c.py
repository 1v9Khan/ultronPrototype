"""Category C -- Security perimeter.

C1  -- firewall inbound-allow / port forwarding / NAT.
C2  -- Defender disable / MsMpEng kill / exclusion adds.
C3  -- installing executables / drivers / cert trust.
C4  -- hosts file / DNS rewrites.
C5  -- reverse shells / tunnels / listening services exposed externally.
C6  -- outbound network calls (curl / iwr) -- LOG_ONLY (legit dev).
C7  -- SSH (already legit; reading ~/.ssh/ privates handled in D1).
C8  -- enabling RDP / WinRM / SMB shares.
C9  -- ``netsh wlan show profile key=clear`` -- Wi-Fi credential dump.
C10 -- system proxy / WPAD / VPN config changes.
C11 -- ARP / DNS client cache manipulation.
C12 -- CDP attach to non-sandbox Chromium profile.
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)
from ultron.safety.validator import Verdict


def build_category_c_rules() -> list[Rule]:
    """Factory for Category C rules."""
    return [
        # C1: firewall inbound / port forwarding / NAT.
        CommandPatternRule(
            rule_id="C1",
            description="firewall inbound-allow / port forwarding / NAT changes",
            category="C",
            patterns=[
                r"\bNew-NetFirewallRule\b.*\bAction\s+Allow\b.*\bInbound\b",
                r"\bSet-NetFirewallRule\b.*\bAction\s+Allow\b",
                r"\bnetsh\s+advfirewall\s+",
                r"\bnetsh\s+firewall\s+(set|add)\s+",
                r"\bnetsh\s+interface\s+portproxy\b",
                # iptables / nftables (Linux / WSL)
                r"\biptables\s+-[AI]\s+INPUT\b.*-j\s+ACCEPT\b",
                r"\bnft\s+add\s+rule\b",
            ],
        ),
        # C2: Defender disable / kill / exclusion adds (specialised in P).
        CommandPatternRule(
            rule_id="C2",
            description="Windows Defender disable / kill / exclusion adds",
            category="C",
            patterns=[
                r"\bAdd-MpPreference\b",
                r"\bSet-MpPreference\b.*-Disable\w+\b",
                r"\bDisable-WindowsOptionalFeature\b.*\bDefender\b",
                r"\bStop-Service\s+(WinDefend|SecurityHealthService|Sense|WdNisSvc)\b",
                r"\bsc(\.exe)?\s+stop\s+(WinDefend|SecurityHealthService|Sense)\b",
                r"\bTaskKill\b.*\bMsMpEng\b",
                r"\bStop-Process\b.*-Name\s+MsMpEng\b",
            ],
        ),
        # C3: installing executables / drivers / cert trust.
        CommandPatternRule(
            rule_id="C3",
            description="installing executables / drivers / cert trust additions",
            category="C",
            patterns=[
                r"\bmsiexec\s+/i\b",
                r"\bAdd-AppxPackage\b",
                r"\bpnputil\s+/add-driver\b",
                r"\bAdd-WindowsDriver\b",
                r"\bcertutil\s+-addstore\b",
                r"\bImport-Certificate\b.*-CertStoreLocation\b",
            ],
        ),
        # C4: hosts file / DNS rewrites.
        PathPatternRule(
            rule_id="C4",
            description="writes to hosts file",
            category="C",
            patterns=[
                r"/system32/drivers/etc/hosts$",
                r"/etc/hosts$",
            ],
        ),
        # C5: reverse shells / tunneling / listening services.
        CommandPatternRule(
            rule_id="C5",
            description="reverse shells / tunneling / externally-exposed listeners",
            category="C",
            patterns=[
                r"\bncat\b.*-[el]\b",
                r"\bnc\b.*-[el]\b",
                r"\bnetcat\b.*-l",
                # ngrok / cloudflared / localtunnel
                r"\bngrok\s+http\b",
                r"\bcloudflared\s+tunnel\b",
                r"\blocaltunnel\b",
                r"\blt\s+--port\b",
                # PowerShell reverse-shell shapes
                r"\$client\s*=\s*New-Object\s+System\.Net\.Sockets\.TCPClient",
                r"\bSystem\.Net\.Sockets\.TcpClient\b.*\bGetStream\b",
                # Common reverse-shell payloads
                r"bash\s+-i\s+>&\s*/dev/tcp/",
            ],
        ),
        # C6: outbound network calls -- LOG_ONLY (legit dev). The
        # actual destination check happens in J-category rules.
        CommandPatternRule(
            rule_id="C6",
            description="outbound HTTP fetch (curl / Invoke-WebRequest / wget)",
            category="C",
            patterns=[
                # We want to LOG_ONLY here; the destination-allowlist
                # rule lives in J6. So pattern-match the fetch and
                # downgrade verdict.
                r"\bcurl\s+(?!.*--help)",
                r"\bwget\s+(?!.*--help)",
                r"\bInvoke-WebRequest\b",
                r"\bInvoke-RestMethod\b",
                r"\biwr\b",
            ],
            verdict_on_match=Verdict.LOG_ONLY,
        ),
        # C8: enabling RDP / WinRM / SMB shares.
        CommandPatternRule(
            rule_id="C8",
            description="enable RDP / WinRM / SMB shares",
            category="C",
            patterns=[
                r"fDenyTSConnections\s*=?\s*0",        # registry-based RDP enable
                r"\bEnable-PSRemoting\b",
                r"\bwinrm\s+quickconfig\b",
                r"\bnet\s+share\s+\w+\s*=",
                r"\bNew-SmbShare\b",
            ],
        ),
        # C9: cleartext Wi-Fi credential dump.
        CommandPatternRule(
            rule_id="C9",
            description="cleartext Wi-Fi credential dump",
            category="C",
            patterns=[
                r"\bnetsh\s+wlan\s+show\s+profile\s+.*key=clear\b",
                r"\bGet-WiFi.*-AsPlainText\b",
            ],
        ),
        # C10: proxy / WPAD / VPN config writes.
        CommandPatternRule(
            rule_id="C10",
            description="system proxy / WPAD / VPN config changes",
            category="C",
            patterns=[
                r"\bnetsh\s+winhttp\s+set\s+proxy\b",
                r"\bSet-ItemProperty\b.*ProxyServer\b",
                r"\bAdd-VpnConnection\b",
                r"\bSet-VpnConnection\b",
                r"Internet Settings\\ProxyServer",
            ],
        ),
        # C11: ARP / DNS client cache manipulation.
        CommandPatternRule(
            rule_id="C11",
            description="ARP table / DNS client cache manipulation",
            category="C",
            patterns=[
                r"\barp\s+-s\s+",
                r"\bAdd-DnsClientCache\b",
                r"\bSet-DnsClient\b.*-ConnectionSpecificSuffix\b",
            ],
        ),
        # C12: Chrome DevTools Protocol attach to user-owned profile.
        CommandPatternRule(
            rule_id="C12",
            description="Chrome DevTools Protocol attach to non-sandbox profile",
            category="C",
            patterns=[
                # Chromium debug-port flags pointed at the user's main
                # profile dir.
                r"--remote-debugging-port=\d+",
                r"--remote-debugging-pipe\b",
                # Direct CDP attach libraries -- block when the URL
                # implies a non-sandbox connect.
                r"chrome_remote_interface",
            ],
        ),
    ]
