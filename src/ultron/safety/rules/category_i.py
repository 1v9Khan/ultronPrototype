"""Category I -- Outbound impact (things that affect others, not the PC).

I1 -- sending emails without explicit "send X to Y" intent.
I2 -- posting to social media accounts.
I3 -- financial / cryptocurrency / online purchases.
I4 -- paid APIs other than the project-approved set.
I5 -- mass messaging on Telegram / Slack / Discord (>5 recipients).
I6 -- YouTube / Twitch / streaming-account API publish actions.
I7 -- cross-capability bridge (browser action verbs on authenticated tabs).
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule
from ultron.safety.validator import Verdict


def build_category_i_rules() -> list[Rule]:
    """Factory for Category I rules."""
    return [
        # I1: send emails without explicit intent.
        CommandPatternRule(
            rule_id="I1",
            description="sending emails without explicit user-stated intent",
            category="I",
            patterns=[
                r"\bSend-MailMessage\b",
                r"\bsmtplib\.SMTP\b.*\.sendmail\b",
                r"\b(?:python-)?msmtp\b",
                r"\bmail\s+-s\b.*<\s*\S+",
                r"\bsendmail\s+-",
                # OpenClaw email tool
                r"\bopenclaw\.message\.send\b.*--channel\s+email\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # I2: social media posts.
        CommandPatternRule(
            rule_id="I2",
            description="posting to social media accounts",
            category="I",
            patterns=[
                # Twitter / X
                r"\b(twitter|tweepy)\.update_status\b",
                r"\bapi\.twitter\.com/2/tweets\b.*POST",
                # Mastodon / Bluesky
                r"\bmastodon\.status_post\b",
                r"\bbsky\.\w+\.createRecord\b",
                # Facebook / Instagram (Meta Graph API)
                r"\bgraph\.facebook\.com\b.*POST",
                # Reddit
                r"\breddit\.\w+\.submit\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # I3: financial / crypto transactions.
        CommandPatternRule(
            rule_id="I3",
            description="financial / cryptocurrency / online purchases",
            category="I",
            patterns=[
                # Crypto wallet send commands
                r"\b(bitcoin|ethereum|web3|solana)-?cli\s+(send|transfer)\b",
                r"\bw3\s+.*\.sendTransaction\b",
                r"\bcontract\.functions\.\w+\(.*\)\.transact\b",
                # Stripe / PayPal / Square API
                r"stripe\.\w+\.PaymentIntent\.create",
                r"paypal\..*\.create_payment",
                r"square\..*\.create_payment",
                # Online shopping fill+submit (heuristic; better to
                # block via Cap-3's action-verb matching).
            ],
        ),
        # I4: paid APIs other than the project-approved set.
        # Phase 2 implements the bare patterns; Phase 5 cross-cutting
        # adds the hostname allowlist check.
        CommandPatternRule(
            rule_id="I4",
            description="calls to paid APIs other than the project-approved set",
            category="I",
            patterns=[
                # OpenAI / Mistral / Cohere / Replicate / fal / etc.
                # (Anthropic is in the allow list -- not blocked.)
                r"\bapi\.openai\.com\b",
                r"\bapi\.mistral\.ai\b",
                r"\bapi\.cohere\.\w+\b",
                r"\bapi\.replicate\.com\b",
                r"\bfal\.run\b",
                r"\bapi\.runwayml\.com\b",
                r"\bapi\.suno\.ai\b",
                r"\bgenerativelanguage\.googleapis\.com\b",
                r"\bapi\.elevenlabs\.io\b",
                r"\bapi\.deepseek\.com\b",
            ],
        ),
        # I5: mass messaging > 5 recipients.
        # Hard to detect without argument parsing; pattern-match the
        # shape that fans out.
        CommandPatternRule(
            rule_id="I5",
            description="mass messaging (>5 recipients in one op)",
            category="I",
            patterns=[
                # Lists of recipients in send_message calls
                r"recipients\s*=\s*\[\s*[\"']\w+[\"']\s*(?:,\s*[\"']\w+[\"']\s*){5,}\]",
                # Bulk Telegram broadcasts
                r"\bsend_message_to_chats\b",
                r"\bbroadcast\b.*chat_ids\s*=",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # I6: streaming-account publish actions.
        CommandPatternRule(
            rule_id="I6",
            description="YouTube / Twitch / streaming publish actions",
            category="I",
            patterns=[
                # YouTube Data API v3 -- upload / schedule / visibility
                r"\byoutube/v3/videos\b.*POST",
                r"\byoutube/v3/liveBroadcasts\b.*POST",
                # Twitch Helix
                r"\bapi\.twitch\.tv/helix/streams\b.*POST",
                r"\bapi\.twitch\.tv/helix/videos\b.*(PATCH|DELETE)",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
    ]
