"""Safe, bounded recovery advice for OpenAI-compatible HTTP 429 responses."""

import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


QUOTA_CODES = {
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}
TRANSIENT_CODES = {"rate_limit_exceeded", "slow_down"}
MAX_RETRIES = 2
MAX_AUTO_WAIT = 15


def retry_after_seconds(value):
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            delay = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None
    return delay if math.isfinite(delay) and delay >= 0 else None


def limit_advice(response, retries):
    """Return a safe user message and optional delay; never expose provider text."""
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        error = error if isinstance(error, dict) else {}
    except ValueError:
        error = {}
    code = error.get("code")
    kind = error.get("type")
    code = code if isinstance(code, str) else None
    kind = kind if isinstance(kind, str) else None
    if code in QUOTA_CODES or kind == "insufficient_quota":
        reasons = {
            "credit_balance_exhausted": "Prepaid API credits are exhausted. Check API billing and add credits before trying again.",
            "project_spend_limit_exceeded": "The project's API spending limit was reached. Check the project's limit before trying again.",
            "organization_spend_limit_exceeded": "The organization's API spending limit was reached. Check the organization's limit before trying again.",
            "organization_usage_limit_exceeded": "The organization's approved API usage limit was reached. Check the organization's usage limits before trying again.",
        }
        if code in reasons:
            return ("Astra's provider returned HTTP 429. " + reasons[code], None)
        return (
            "Astra's provider reports exhausted quota, credits, or a spending limit (HTTP 429). "
            "Check API billing and the project's or organization's limits before trying again.",
            None,
        )
    delay = retry_after_seconds(response.headers.get("Retry-After"))
    transient = code in TRANSIENT_CODES or kind == "rate_limit_error"
    if not transient and delay is None:
        return (
            "Astra's provider returned HTTP 429 without a recognizable retry hint. "
            "This can indicate a rate limit or exhausted quota. Check API usage and billing before trying again.",
            None,
        )
    if delay is not None and delay > MAX_AUTO_WAIT:
        return (f"Astra's provider returned HTTP 429; wait at least {math.ceil(delay)} seconds before trying again.", None)
    if retries >= MAX_RETRIES:
        wait = f" Wait at least {math.ceil(delay)} seconds before trying again." if delay is not None else " Try again later."
        return ("Astra's provider rate limit (HTTP 429) persists after two retries." + wait, None)
    if delay is None:
        delay = 2 ** retries + random.uniform(0, 0.5)
    return (f"Astra's provider is rate limited (HTTP 429); retrying in {delay:.1f} seconds ({retries + 1}/{MAX_RETRIES}).", delay)
