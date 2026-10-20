"""Document parsing module using LLM for structured field extraction."""

import json
import logging
import os
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _is_local_base(url: Optional[str]) -> bool:
    """True when a base_url points at a loopback / local container host.

    Local OpenAI-compatible servers (Ollama, LM Studio, llama.cpp, vLLM) run
    without an API key, so for these hosts we may proceed keylessly.
    """
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return (
        host in {"localhost", "0.0.0.0", "::1", "127.0.0.1", "host.docker.internal", "gateway.docker.internal"}
        or host.startswith("127.")
        or host.endswith(".local")
        or host.endswith(".internal")
    )


def _keyless_allowed(url: Optional[str]) -> bool:
    """Allow an empty API key for local endpoints, or when explicitly opted in
    via OCR_LLM_KEYLESS=true (e.g. a trusted private-LAN server)."""
    if _is_local_base(url):
        return True
    return (os.getenv("OCR_LLM_KEYLESS") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

# Default prompt for document parsing
DEFAULT_PARSE_PROMPT = """You are a document parser. Given the following OCR text extracted from an image, 
extract structured fields. Return a JSON object with the fields you can identify.

Common fields to look for: name, date, address, phone, email, amount, total, invoice_number, 
company, description, quantity, price, due_date, etc.

OCR Text:
{text}

Return ONLY valid JSON. If a field is not found, omit it."""


def _build_field_prompt(fields: list[str]) -> str:
    """Build a parser prompt that returns EXACTLY the requested field names.

    Lets a caller describe its own form so the LLM fills those specific fields
    instead of guessing a generic set. Field names are sanitised so they cannot
    break the ``{text}`` .format() substitution.
    """
    safe = [str(f).strip().strip("{}") for f in fields if str(f).strip("{}")]
    field_list = ", ".join(safe)
    return (
        "You are a document parser. Extract these specific fields from the OCR text "
        "so they can populate a form.\n\n"
        f"Fields to extract: {field_list}\n\n"
        "Return ONLY a single JSON object whose top-level keys are EXACTLY these "
        f"and nothing else: {field_list}.\n"
        "Rules:\n"
        "- The field names are hints, NOT necessarily the document's wording. Match each "
        "field to wherever it appears under an equivalent label, treating underscores as "
        "spaces (e.g. 'total_due' may appear as 'TOTAL DUE' or 'Amount Due'; 'bill_to' as "
        "'Bill to'; 'invoice_number' as 'Invoice #' or 'Inv No').\n"
        "- Only return null for a field whose value genuinely is not present in the text.\n"
        "- For amount/total fields, return the FINAL amount owed, not a line item or subtotal.\n"
        "- Format dates as YYYY-MM-DD.\n"
        "- Write amounts/quantities as plain numbers without currency symbols or "
        "thousands separators.\n"
        "- If a field is clearly a list (e.g. line items), use a JSON array as its value.\n"
        "- Do not invent data; copy values as they appear in the document.\n\n"
        "OCR Text:\n{text}\n\n"
        "Return ONLY valid JSON."
    )


def parse_document(
    text: str,
    prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    fields: Optional[list] = None,
) -> Optional[dict]:
    """
    Parse OCR text into structured fields using an LLM.

    Supports any OpenAI-compatible API (OpenAI, OpenRouter, local models, ...).
    Set environment variables:
    - OPENAI_API_KEY or OCR_LLM_API_KEY
    - OPENAI_BASE_URL or OCR_LLM_BASE_URL (e.g. https://openrouter.ai/api/v1)
    - OCR_LLM_MODEL (or OPENAI_MODEL) to choose the model id
    - OCR_LLM_TITLE / OCR_LLM_REFERER (optional, OpenRouter attribution headers)

    Local / self-hosted LLMs (Ollama, LM Studio, llama.cpp, vLLM) expose an
    OpenAI-compatible endpoint. Ollama needs no key: just set OCR_LLM_BASE_URL
    (e.g. http://localhost:11434/v1) and OCR_LLM_MODEL. LM Studio's server has an
    optional API-key toggle — if it's on, set OCR_LLM_API_KEY to that key (it is
    sent as Authorization: Bearer); if it's off, leave the key empty. Keyless
    access (empty key) is allowed automatically for loopback / host.docker.internal
    / *.local hosts; for a trusted private-LAN server set OCR_LLM_KEYLESS=true.

    Args:
        text: The OCR-extracted text.
        prompt: Custom prompt template (use {text} placeholder).
        api_key: API key override.
        base_url: Base URL override for OpenAI-compatible endpoints.
        model: Model name override. Defaults to OCR_LLM_MODEL/OPENAI_MODEL env,
            then 'gpt-4o-mini'. OpenRouter ids are namespaced (e.g. 'openai/gpt-4o-mini').

    Returns:
        Parsed dict of structured fields, or None if LLM not configured.
    """
    key = api_key or os.getenv("OCR_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("OCR_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not key:
        if url and _keyless_allowed(url):
            # Local OpenAI-compatible servers ignore the key; the SDK still
            # requires a non-empty string, so send a harmless placeholder.
            key = "not-needed"
        else:
            logger.warning(
                "No LLM API key configured and no local/keyless endpoint detected "
                "(set OCR_LLM_BASE_URL to a local server, or OCR_LLM_KEYLESS=true for a "
                "trusted LAN host). Skipping document parsing."
            )
            return None

    model = model or os.getenv("OCR_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    try:
        max_tokens = int(os.getenv("OCR_LLM_MAX_TOKENS") or "4000")
    except (TypeError, ValueError):
        max_tokens = 4000
    try:
        max_input = int(os.getenv("OCR_LLM_MAX_INPUT_CHARS") or "40000")
    except (TypeError, ValueError):
        max_input = 40000
    # Cap input to protect the context window, but preserve the END of the
    # document (summaries/totals usually live there) rather than slicing it off.
    text = text or ""
    if len(text) > max_input:
        head = (max_input * 3) // 4
        tail = max_input - head
        text = text[:head] + "\n[...truncated...]\n" + text[-tail:]
    if prompt:
        parse_prompt = prompt
    elif fields:
        parse_prompt = _build_field_prompt(list(fields))
    else:
        parse_prompt = DEFAULT_PARSE_PROMPT

    try:
        from openai import OpenAI

        client_kwargs = {"api_key": key}
        if url:
            client_kwargs["base_url"] = url
            # OpenRouter recommends optional attribution headers.
            if "openrouter.ai" in url:
                headers = {}
                if os.getenv("OCR_LLM_REFERER"):
                    headers["HTTP-Referer"] = os.getenv("OCR_LLM_REFERER")
                if os.getenv("OCR_LLM_TITLE"):
                    headers["X-Title"] = os.getenv("OCR_LLM_TITLE")
                if headers:
                    client_kwargs["default_headers"] = headers

        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a document parser. Return only valid JSON."},
                {"role": "user", "content": parse_prompt.format(text=text)},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )

        choice = response.choices[0] if response.choices else None
        msg = choice.message if choice else None
        # Reasoning/thinking models can return content=None when the token budget
        # is exhausted before the final answer, so guard against a crash here.
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            finish = getattr(choice, "finish_reason", None) if choice else None
            logger.error(
                "LLM returned no content (finish_reason=%s). The model likely spent "
                "its token budget on reasoning; raise OCR_LLM_MAX_TOKENS or use a "
                "non-reasoning model.", finish,
            )
            return None
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)
        return parsed

    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        return None
