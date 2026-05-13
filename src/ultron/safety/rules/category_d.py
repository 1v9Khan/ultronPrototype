"""Category D -- Credential / secret access.

Principle: the model never sees credential bytes. Apps the model
launches CAN use credentials (Chrome with default profile); the
model reading the credential files directly is what's blocked.

These are mostly OUT-gate rules: ``write_only=False`` since the
threat is READS that exfiltrate the bytes into the model's context.

D1  -- ~/.ssh/id_*, ~/.aws/credentials, GPG, Credential Manager.
D2  -- browser profile binary credential files.
D3  -- LSASS dump / cross-process memory inspection.
D4  -- certificate store private keys.
D5  -- crypto wallet files.
D6  -- .env files outside the project.
D7  -- streaming any of the above into LLM context.
D8  -- cloud CLI credential dirs (.azure, gcloud, kube, docker, gh).
D9  -- package-manager auth tokens (.npmrc, .pypirc, etc.).
D10 -- DPAPI master keys / Kerberos ticket cache / SAM hives.
D11 -- password manager databases (KeePass, 1P, Bitwarden).
D12 -- mail-store files (Outlook .pst/.ost, Thunderbird).
D13 -- clipboard reads (block by default; explicit-intent override).
D15 -- keylogging via low-level hooks.
D16 -- microphone / webcam access outside voice pipeline.
D17 -- browser session SQLite direct reads.
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)
from ultron.safety.validator import Verdict


def build_category_d_rules() -> list[Rule]:
    """Factory for Category D rules.

    Note: many D rules cover paths outside PROJECT_ROOT (user's home
    directory, browser profile dirs in %LOCALAPPDATA%). The
    PathPatternRule matches the canonical (lowercase, forward-slash)
    form which works across Windows path drive prefixes.
    """
    return [
        # D1: SSH / AWS / GPG / Credential Manager paths.
        PathPatternRule(
            rule_id="D1",
            description="SSH / AWS / GPG / Windows Credential Manager files",
            category="D",
            patterns=[
                r"/\.ssh/id_\w+($|[^/])",
                r"/\.ssh/id_\w+\.pub$",  # public keys aren't sensitive but
                # reading them en masse implies fingerprinting; block by
                # default and the user can carve out.
                r"/\.aws/credentials$",
                r"/\.aws/config$",
                r"/\.gnupg/(secring|private-keys-v1\.d)/",
                r"/microsoft/credentials/",
                r"/microsoft/vault/",
            ],
            write_only=False,
        ),
        # D2: browser profile binary credential files.
        PathPatternRule(
            rule_id="D2",
            description="browser profile credential binary files",
            category="D",
            patterns=[
                # Chrome / Edge / Brave / Opera
                r"/(google/chrome|microsoft/edge|brave-?software|opera software)/user data/(default|profile \d+)/(login data|cookies|web data|local state)$",
                # Firefox
                r"/mozilla/firefox/profiles/[^/]+/(logins\.json|cookies\.sqlite|key\d?\.db|signons\.sqlite)$",
            ],
            write_only=False,
        ),
        # D3: LSASS dump / cross-process memory inspection.
        CommandPatternRule(
            rule_id="D3",
            description="LSASS dump / cross-process memory inspection",
            category="D",
            patterns=[
                r"\blsass\.exe\b",
                r"\bMiniDumpWriteDump\b",
                r"\bcomsvcs\.dll\s+MiniDump\b",
                r"\bprocdump.*lsass\b",
                r"\bOpenProcess\b.*lsass",
                # Mimikatz indicators
                r"\bsekurlsa::",
                r"\blsadump::",
                r"\bmimikatz\b",
            ],
        ),
        # D4: certificate store private keys.
        PathPatternRule(
            rule_id="D4",
            description="certificate store private keys",
            category="D",
            patterns=[
                r"/microsoft/crypto/",
                r"/microsoft/protect/",
                # `cert:\` PowerShell drive accessed via paths
                r"\bcert:\\\\(currentuser|localmachine)\\\\my\\\\",
            ],
            write_only=False,
        ),
        # D4 (also): cert export commands
        CommandPatternRule(
            rule_id="D4",
            description="exporting certificate private keys",
            category="D",
            patterns=[
                r"\bExport-PfxCertificate\b",
                r"\bcertutil\s+-(exportPFX|dump.*-p\s+)",
            ],
        ),
        # D5: crypto wallet files.
        PathPatternRule(
            rule_id="D5",
            description="cryptocurrency wallet files",
            category="D",
            patterns=[
                r"\.wallet$",
                r"/wallet\.dat$",
                r"/keystore/[^/]+\.json$",     # Ethereum / MetaMask keystore
                r"/electrum/wallets/",
                r"/metamask/.*\.json$",
                r"/bitcoin/wallet\.dat",
                r"/litecoin/wallet\.dat",
            ],
            write_only=False,
        ),
        # D6: .env files outside the project.
        # The user's restriction list marks this ○ -- block by default,
        # allow inside PROJECT_ROOT. Implemented as a path pattern that
        # checks for ``.env`` files NOT under the project's known dirs.
        # The pattern fires on any .env path; the rule's interpretation
        # is that PROJECT_ROOT-relative paths get caught upstream by
        # the project's own safety/explicit-intent logic. For Phase 2
        # we treat all .env reads as NEEDS_EXPLICIT_INTENT to err safe.
        PathPatternRule(
            rule_id="D6",
            description=".env file reads outside the project",
            category="D",
            patterns=[
                # .env at any path; the validator's outer ACL allows
                # PROJECT_ROOT/.env explicitly via Category K's protected
                # list (which is about WRITES; reads aren't blocked at K).
                # For now, NEEDS_EXPLICIT_INTENT means the user has to
                # have explicitly asked.
                r"/\.env(\.[a-z]+)?$",
                r"/\.env\.local$",
                r"/\.env\.production$",
            ],
            write_only=False,
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # D8: cloud CLI credential directories.
        PathPatternRule(
            rule_id="D8",
            description="cloud CLI credential directories",
            category="D",
            patterns=[
                r"/\.azure/(accessTokens|azureProfile)\.json$",
                r"/\.config/gcloud/(credentials|application_default_credentials)\.json$",
                r"/\.kube/config$",
                r"/\.docker/config\.json$",
                r"/\.config/gh/hosts\.yml$",
                r"/\.config/hub$",
            ],
            write_only=False,
        ),
        # D9: package-manager auth tokens.
        PathPatternRule(
            rule_id="D9",
            description="package-manager auth tokens",
            category="D",
            patterns=[
                r"/\.npmrc$",
                r"/\.pypirc$",
                r"/\.cargo/credentials(\.toml)?$",
                r"/\.gem/credentials$",
                r"/\.nuget/nuget\.config$",
            ],
            write_only=False,
        ),
        # D10: DPAPI master keys / Kerberos / SAM hives.
        PathPatternRule(
            rule_id="D10",
            description="DPAPI master keys / SAM hives / Kerberos tickets",
            category="D",
            patterns=[
                r"/microsoft/protect/",
                r"/microsoft/credentials/",
                r"/microsoft/vault/",
                r"/system32/config/(sam|system|security)($|\.\w+)",
                r"\.kirbi$",                # Kerberos ticket cache
                r"\.ccache$",
            ],
            write_only=False,
        ),
        # D10 (also): commands that read these
        CommandPatternRule(
            rule_id="D10",
            description="raw SAM / SECURITY hive dump commands",
            category="D",
            patterns=[
                r"\breg\s+save\s+hklm\\\\(sam|security|system)\b",
                r"\bGet-WmiObject\b.*Win32_PnPSignedDriver\b",
                r"\bsecretsdump\b",
                r"\bklist\s+tickets\b",
            ],
        ),
        # D11: password manager databases.
        PathPatternRule(
            rule_id="D11",
            description="password-manager databases",
            category="D",
            patterns=[
                r"\.kdbx$",
                r"\.kdb$",
                r"/1password( \d)?/data/",
                r"/bitwarden/data\.json$",
                r"/lastpass/",
                r"/dashlane/",
                r"/keeper/",
            ],
            write_only=False,
        ),
        # D12: mail stores.
        PathPatternRule(
            rule_id="D12",
            description="mail-store files (Outlook PST/OST, Thunderbird mbox)",
            category="D",
            patterns=[
                r"\.pst$",
                r"\.ost$",
                r"/thunderbird/profiles/[^/]+/imapmail/",
                r"/thunderbird/profiles/[^/]+/mail/",
            ],
            write_only=False,
        ),
        # D13: clipboard reads.
        CommandPatternRule(
            rule_id="D13",
            description="clipboard reads (passwords copied seconds ago)",
            category="D",
            patterns=[
                r"\bGet-Clipboard\b",
                r"\bPaste-Clipboard\b",
                r"\bxclip\s+-selection\s+clipboard\s+-o\b",
                r"\bxsel\s+--clipboard\s+--output\b",
                r"\bpbpaste\b",
                r"\bclipboard\.GetText\b",
                r"\bpyperclip\.paste\b",
                r"\btkinter\..*\.clipboard_get\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # D15: keylogging via low-level hooks.
        CommandPatternRule(
            rule_id="D15",
            description="keylogging via low-level hooks / raw input",
            category="D",
            patterns=[
                r"\bSetWindowsHookEx\b.*\bWH_KEYBOARD_LL\b",
                r"\bGetAsyncKeyState\b.*loop",
                r"\bRegisterRawInputDevices\b",
                # Python keyloggers
                r"\bpynput\.keyboard\.Listener\b",
                r"\bkeyboard\.on_press\b",
            ],
        ),
        # D16: microphone / webcam outside the voice pipeline.
        CommandPatternRule(
            rule_id="D16",
            description="microphone / webcam access outside voice pipeline",
            category="D",
            patterns=[
                # Webcam capture
                r"\bcv2\.VideoCapture\b",
                r"\bImage\.Capture\b",
                r"\bVideoSource\b",
                # Mic capture outside the Whisper / Silero / Smart Turn
                # pipeline. These pipelines use sounddevice via
                # AudioCapture; ad-hoc PyAudio reads in OTHER tools
                # are blocked.
                r"\bPyAudio\(\)\.open\b",   # Note: AudioCapture uses sounddevice
                r"\bnAudio\.WaveIn\b",
            ],
        ),
        # D17: browser session SQLite direct reads.
        PathPatternRule(
            rule_id="D17",
            description="browser session SQLite direct reads",
            category="D",
            patterns=[
                # Same as D2 but explicit for the SQLite files
                r"/(default|profile \d+)/cookies(-journal)?$",
                r"/firefox/profiles/[^/]+/cookies\.sqlite$",
            ],
            write_only=False,
        ),
    ]
