"""API-key authentication and admin key management for the SaaS deployment.

Design:
- Clients authenticate to the OCR endpoints with a per-client API key sent in
  the ``X-API-Key`` header (``Authorization: Bearer <key>`` also accepted).
- Keys are stored hashed (SHA-256) in a small SQLite DB, so a leaked DB does
  not expose usable keys. The raw key is only ever shown once, at creation.
- Keys are issued/revoked through the ``/admin/keys`` endpoints, protected by
  a separate long-lived admin token supplied via the ``X-Admin-Token`` header
  and configured with the ``OCR_ADMIN_TOKEN`` environment variable.
- No third-party dependencies for authentication: stdlib ``sqlite3`` + ``secrets``.
  The optional per-client LLM credentials stored by the admin portal are
  encrypted with ``cryptography`` (Fernet); see ``update_llm_config``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader

from .schemas import (
    APIKeyCreateRequest,
    APIKeyCreated,
    APIKeyInfo,
    APIKeyList,
    APIKeyUpdateRequest,
    LLMConfigUpdate,
    UsagePoint,
    UsageSummary,
    UsageReport,
    ClientProfile,
    AdminSummary,
    LoginRequest,
    SignupRequest,
    DemoLoginRequest,
    AuthUser,
    AdminPublic,
    ClientAccount,
    SignupResponse,
    OcrRun,
    OcrHistory,
)

logger = logging.getLogger(__name__)

# --- Configuration (env-driven, with safe defaults) ------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("OCR_AUTH_DB", str(PROJECT_ROOT / "auth.db")))
# Master-key file for the encrypted per-client LLM credentials. Used only when
# OCR_SECRET_KEY is not set; kept next to the DB so backups keep both together.
LLM_KEY_FILE = Path(os.getenv("OCR_LLM_KEY_FILE", str(DB_PATH.parent / "llm.key")))
ADMIN_TOKEN = os.getenv("OCR_ADMIN_TOKEN", "").strip()
BOOTSTRAP_API_KEY = os.getenv("OCR_BOOTSTRAP_API_KEY", "").strip()
# Defaults applied to newly created keys when the admin does not set them.
# Empty -> unlimited. Useful for "every new customer starts on plan X".
DEFAULT_MONTHLY_QUOTA = os.getenv("OCR_DEFAULT_MONTHLY_QUOTA", "").strip()
DEFAULT_RPM_LIMIT = os.getenv("OCR_DEFAULT_RPM_LIMIT", "").strip()
DEFAULT_PLAN = os.getenv("OCR_DEFAULT_PLAN", "").strip()
# 'hard' (default) blocks with 429 when monthly quota is hit; 'soft' keeps
# serving but flags the response. Applies to new keys unless overridden.
DEFAULT_OVER_LIMIT_MODE = os.getenv("OCR_DEFAULT_OVER_LIMIT_MODE", "").strip()

# Seed the first admin account on startup when the admins table is empty.
ADMIN_EMAIL = os.getenv("OCR_ADMIN_EMAIL", "").strip().lower()
ADMIN_PASSWORD = os.getenv("OCR_ADMIN_PASSWORD", "").strip()

# Demo mode: exposes POST /auth/demo, which seeds/refreshes a fixed demo admin
# and demo client account and logs you straight in. DISABLED BY DEFAULT -- only
# turn it on (OCR_DEMO_MODE=true) for local demos, never on a public deployment.
DEMO_MODE = os.getenv("OCR_DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")
DEMO_ADMIN_EMAIL = os.getenv("OCR_DEMO_ADMIN_EMAIL", "admin@example.com").strip().lower()
DEMO_ADMIN_PASSWORD = os.getenv("OCR_DEMO_ADMIN_PASSWORD", "change-me-please")
DEMO_CLIENT_EMAIL = os.getenv("OCR_DEMO_CLIENT_EMAIL", "demo@client.com").strip().lower()
DEMO_CLIENT_PASSWORD = os.getenv("OCR_DEMO_CLIENT_PASSWORD", "demo12345")
DEMO_CLIENT_NAME = os.getenv("OCR_DEMO_CLIENT_NAME", "Demo Client")

# Session cookies / tokens.
SESSION_COOKIE = os.getenv("OCR_SESSION_COOKIE", "ocr_session").strip()
ADMIN_SESSION_TTL_SECONDS = int(os.getenv("OCR_ADMIN_SESSION_TTL_HOURS", "12")) * 3600
CLIENT_SESSION_TTL_SECONDS = int(os.getenv("OCR_CLIENT_SESSION_TTL_DAYS", "7")) * 86400
# Set OCR_COOKIE_SECURE=true when serving behind HTTPS so the cookie is TLS-only.
COOKIE_SECURE = os.getenv("OCR_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes", "on")

_KEY_PREFIX_LEN = 12  # how much of the raw key we keep (non-secret) for display

# --- In-memory rate limiting (per-process, per-minute sliding window) ------
# NOTE: this is per-worker-process. The service is designed to run with a
# single uvicorn worker (PaddleOCR models are cached per process), so this is
# accurate in the intended deployment. Cross-process limits would need Redis.
_rate_lock = threading.Lock()
_rate_windows: dict[int, deque] = {}

# OpenAPI security schemes -> powers the "Authorize" button in /docs
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_header = APIKeyHeader(name="Authorization", auto_error=False, scheme_name="Bearer")
_admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


# --- Low-level storage -----------------------------------------------------
@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash      TEXT UNIQUE NOT NULL,
                key_prefix    TEXT NOT NULL,
                name          TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                last_used_at  TEXT,
                request_count INTEGER NOT NULL DEFAULT 0,
                is_active     INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_daily (
                key_id INTEGER NOT NULL,
                day    TEXT NOT NULL,       -- YYYY-MM-DD (UTC)
                count  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key_id, day)
            );
            """
        )
        _ensure_quota_columns(conn)
        _ensure_llm_columns(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admins (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                pw_hash       TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                last_login_at TEXT,
                is_active     INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clients (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                pw_hash       TEXT NOT NULL,
                key_id        INTEGER NOT NULL,
                created_at    TEXT NOT NULL,
                last_login_at TEXT,
                is_active     INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash   TEXT PRIMARY KEY,
                principal    TEXT NOT NULL,      -- 'admin' | 'client'
                principal_id INTEGER NOT NULL,
                created_at   TEXT NOT NULL,
                expires_at   TEXT NOT NULL,
                revoked_at   TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_principal ON sessions (principal, principal_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id          INTEGER NOT NULL,
                created_at      TEXT NOT NULL,
                filename        TEXT,
                document_type   TEXT,
                lang            TEXT,
                text_blocks     INTEGER NOT NULL DEFAULT 0,
                tables          INTEGER NOT NULL DEFAULT 0,
                pages           INTEGER NOT NULL DEFAULT 0,
                chars           INTEGER NOT NULL DEFAULT 0,
                files_count     INTEGER NOT NULL DEFAULT 1,
                table_detection INTEGER NOT NULL DEFAULT 0,
                llm_parsing     INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocr_runs_key ON ocr_runs (key_id, created_at)"
        )
        _ensure_ocr_run_columns(conn)


def _ensure_quota_columns(conn: sqlite3.Connection) -> None:
    """Add plan/quota/rate columns to an existing api_keys table (idempotent)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(api_keys)").fetchall()}
    for name, sql_type in (
        ("plan", "TEXT"),
        ("monthly_quota", "INTEGER"),
        ("rpm_limit", "INTEGER"),
        ("over_limit_mode", "TEXT"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE api_keys ADD COLUMN {name} {sql_type}")


def _ensure_llm_columns(conn: sqlite3.Connection) -> None:
    """Add per-client LLM credential columns to an existing api_keys table."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(api_keys)").fetchall()}
    for name, sql_type in (
        ("llm_base_url", "TEXT"),
        ("llm_model", "TEXT"),
        ("llm_api_key_enc", "TEXT"),   # Fernet token, never the plaintext key
        ("llm_key_hint", "TEXT"),      # last 4 chars, for display only
        ("llm_updated_at", "TEXT"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE api_keys ADD COLUMN {name} {sql_type}")


def _ensure_ocr_run_columns(conn: sqlite3.Connection) -> None:
    """Add newer columns to an existing ocr_runs table (idempotent)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(ocr_runs)").fetchall()}
    if "files_count" not in cols:
        conn.execute("ALTER TABLE ocr_runs ADD COLUMN files_count INTEGER NOT NULL DEFAULT 1")


def _hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_mode(value: Optional[str]) -> str:
    """Normalise an over-limit mode to 'hard' or 'soft' (default 'hard')."""
    return "soft" if (value or "").strip().lower() == "soft" else "hard"


def _row_to_info(row: sqlite3.Row) -> APIKeyInfo:
    keys = row.keys()
    return APIKeyInfo(
        id=row["id"],
        name=row["name"],
        key_prefix=row["key_prefix"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        request_count=row["request_count"],
        is_active=bool(row["is_active"]),
        plan=(row["plan"] if "plan" in keys else None),
        monthly_quota=(row["monthly_quota"] if "monthly_quota" in keys else None),
        rpm_limit=(row["rpm_limit"] if "rpm_limit" in keys else None),
        over_limit_mode=_norm_mode(row["over_limit_mode"] if "over_limit_mode" in keys else None),
        llm_base_url=(row["llm_base_url"] if "llm_base_url" in keys else None),
        llm_model=(row["llm_model"] if "llm_model" in keys else None),
        llm_key_hint=(row["llm_key_hint"] if "llm_key_hint" in keys else None),
        llm_updated_at=(row["llm_updated_at"] if "llm_updated_at" in keys else None),
    )


# --- Key lifecycle ---------------------------------------------------------
def generate_api_key() -> str:
    """Create a new high-entropy API key string."""
    return "ocr_" + secrets.token_urlsafe(32)


def _default_quota_fields() -> dict:
    def _to_int(v: str) -> Optional[int]:
        return int(v) if v.isdigit() else None
    return {
        "plan": DEFAULT_PLAN or None,
        "monthly_quota": _to_int(DEFAULT_MONTHLY_QUOTA),
        "rpm_limit": _to_int(DEFAULT_RPM_LIMIT),
        "over_limit_mode": _norm_mode(DEFAULT_OVER_LIMIT_MODE),
    }


def create_api_key(
    name: str,
    monthly_quota: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    plan: Optional[str] = None,
    over_limit_mode: Optional[str] = None,
) -> tuple[str, APIKeyInfo]:
    """Issue a new key. Returns (raw_key, record). The raw key is shown only once.

    Any unset limit falls back to the deployment default (env-driven).
    """
    defaults = _default_quota_fields()
    if monthly_quota is None:
        monthly_quota = defaults["monthly_quota"]
    if rpm_limit is None:
        rpm_limit = defaults["rpm_limit"]
    if plan is None:
        plan = defaults["plan"]
    mode = _norm_mode(over_limit_mode if over_limit_mode is not None else defaults["over_limit_mode"])
    raw = generate_api_key()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, name, created_at, plan, monthly_quota, rpm_limit, over_limit_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_hash(raw), raw[:_KEY_PREFIX_LEN], name, _now(), plan, monthly_quota, rpm_limit, mode),
        )
        row = conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    logger.info("Created API key id=%s name=%r quota=%s rpm=%s plan=%s mode=%s",
                row["id"], name, monthly_quota, rpm_limit, plan, mode)
    return raw, _row_to_info(row)


def update_api_key(key_id: int, fields: dict) -> Optional[APIKeyInfo]:
    """Update the given fields of a key. A value of None clears that column.

    Only keys present in ``fields`` are written (so callers can distinguish
    'leave unchanged' from 'set to unlimited/none').
    """
    allowed = {"name", "plan", "monthly_quota", "rpm_limit", "is_active", "over_limit_mode"}
    cols = {k: v for k, v in fields.items() if k in allowed}
    if not cols:
        return get_api_key(key_id)
    if "over_limit_mode" in cols and cols["over_limit_mode"] is not None:
        cols["over_limit_mode"] = _norm_mode(cols["over_limit_mode"])
    if "is_active" in cols and cols["is_active"] is not None:
        cols["is_active"] = 1 if cols["is_active"] else 0
    assignments = ", ".join(f"{k} = ?" for k in cols)
    values = list(cols.values()) + [key_id]
    with _conn() as conn:
        cur = conn.execute(
            f"UPDATE api_keys SET {assignments} WHERE id = ?", values
        )
        if cur.rowcount == 0:
            return None
    return get_api_key(key_id)


def list_api_keys() -> list[APIKeyInfo]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM api_keys ORDER BY id").fetchall()
    return [_row_to_info(r) for r in rows]


# --- Per-client LLM credentials (admin-managed, encrypted at rest) --------
_UNSET = object()
_fernet_lock = threading.Lock()
_fernet_cipher = None
_fernet_failed = False


def _fernet():
    """Return the Fernet cipher used for stored client LLM keys, or None.

    The master key comes from OCR_SECRET_KEY when set. Otherwise a random key
    is generated once and kept next to the database (llm.key); that file is what
    makes stored credentials decryptable, so it must be backed up with the DB
    and never committed.
    """
    global _fernet_cipher, _fernet_failed
    if _fernet_cipher is not None:
        return _fernet_cipher
    if _fernet_failed:
        return None
    with _fernet_lock:
        if _fernet_cipher is not None:
            return _fernet_cipher
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            logger.error(
                "cryptography is not installed, so per-client LLM keys cannot be stored "
                "(pip install cryptography)."
            )
            _fernet_failed = True
            return None
        secret = os.getenv("OCR_SECRET_KEY", "").strip()
        if secret:
            key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
        else:
            try:
                if LLM_KEY_FILE.exists():
                    key = LLM_KEY_FILE.read_bytes().strip()
                else:
                    LLM_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
                    key = Fernet.generate_key()
                    LLM_KEY_FILE.write_bytes(key)
                    try:
                        LLM_KEY_FILE.chmod(0o600)
                    except OSError:
                        pass
                    logger.warning(
                        "No OCR_SECRET_KEY set; generated %s to encrypt stored client "
                        "LLM keys. Back it up with the database.", LLM_KEY_FILE,
                    )
            except OSError as e:
                logger.error("Cannot read or create %s: %s", LLM_KEY_FILE, e)
                _fernet_failed = True
                return None
        try:
            _fernet_cipher = Fernet(key)
        except Exception as e:  # malformed key file / foreign key material
            logger.error("Invalid LLM credential key (%s): %s", LLM_KEY_FILE, e)
            _fernet_failed = True
            return None
        return _fernet_cipher


def _clean_base_url(value: Optional[str]) -> Optional[str]:
    """Validate an admin-supplied LLM endpoint. None or blank clears it."""
    url = (value or "").strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="LLM base URL must be an http(s) URL, e.g. https://openrouter.ai/api/v1",
        )
    return url.rstrip("/")


def update_llm_config(
    key_id: int,
    base_url=_UNSET,
    model=_UNSET,
    api_key=_UNSET,
) -> Optional[APIKeyInfo]:
    """Set or clear a client's own LLM endpoint/model/key.

    Only the arguments actually passed are written, so omitted fields keep their
    value. The API key is encrypted before it reaches the database and only its
    last four characters are stored in the clear, for display.
    """
    cols: dict = {}
    if base_url is not _UNSET:
        cols["llm_base_url"] = _clean_base_url(base_url)
    if model is not _UNSET:
        cols["llm_model"] = (model or "").strip() or None
    if api_key is not _UNSET:
        raw_key = (api_key or "").strip()
        if raw_key:
            cipher = _fernet()
            if cipher is None:
                raise HTTPException(
                    status_code=503,
                    detail="Server cannot encrypt client LLM keys: install the 'cryptography' package and retry.",
                )
            cols["llm_api_key_enc"] = cipher.encrypt(raw_key.encode("utf-8")).decode("ascii")
            cols["llm_key_hint"] = raw_key[-4:]
        else:
            cols["llm_api_key_enc"] = None
            cols["llm_key_hint"] = None
    if not cols:
        return get_api_key(key_id)
    cols["llm_updated_at"] = _now()
    assignments = ", ".join(f"{name} = ?" for name in cols)
    with _conn() as conn:
        cur = conn.execute(
            f"UPDATE api_keys SET {assignments} WHERE id = ?", [*cols.values(), key_id]
        )
        if cur.rowcount == 0:
            return None
    logger.info("Updated LLM settings for key id=%s (fields=%s)", key_id, sorted(cols))
    return get_api_key(key_id)


def get_llm_override(key_id: int) -> Optional[dict]:
    """Decrypted per-client LLM settings to pass to ``parse_document``.

    Returns a subset of {api_key, base_url, model}, or None when the client
    should use the server-wide OCR_LLM_* configuration. Values are never logged.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT llm_base_url, llm_model, llm_api_key_enc FROM api_keys WHERE id = ?",
            (key_id,),
        ).fetchone()
    if row is None:
        return None
    base_url, model, enc = row["llm_base_url"], row["llm_model"], row["llm_api_key_enc"]
    if not (base_url or model or enc):
        return None
    api_key = None
    if enc:
        cipher = _fernet()
        if cipher is None:
            logger.error("Cannot decrypt the stored LLM key for key id=%s (encryption unavailable).", key_id)
            return None
        try:
            api_key = cipher.decrypt(enc.encode("ascii")).decode("utf-8")
        except Exception as e:
            logger.error("Cannot decrypt the stored LLM key for key id=%s: %s", key_id, e)
            return None
    override: dict = {}
    if base_url:
        override["base_url"] = base_url
    if model:
        override["model"] = model
    if api_key:
        override["api_key"] = api_key
    return override or None


def revoke_api_key(key_id: int) -> bool:
    with _conn() as conn:
        # Drop the encrypted LLM credential too: a revoked client must not leave a
        # billable provider key behind in the database.
        cur = conn.execute(
            "UPDATE api_keys SET is_active = 0, llm_api_key_enc = NULL, llm_key_hint = NULL "
            "WHERE id = ? AND is_active = 1",
            (key_id,),
        )
        return cur.rowcount > 0


def rotate_api_key(key_id: int) -> Optional[tuple[str, APIKeyInfo]]:
    """Replace a key with a fresh one and revoke the old key.

    The raw replacement is returned only here, exactly like create_api_key, so
    this is the supported answer to "the client lost their key" - only hashes
    are stored, so the original cannot be re-displayed.

    Carried over: name, plan, quota, rate limit, over-limit mode and the
    per-client LLM credentials (revoking would wipe those). Usage rows
    (usage_daily), OCR history (ocr_runs) and any client account bound to the key
    are re-pointed at the new row, so a rotation neither resets this month's
    quota nor hides past runs. The old key stops working immediately.

    Returns (raw_key, new_record), or None when the key is unknown/revoked.
    """
    raw = generate_api_key()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        if row is None or not row["is_active"]:
            return None
        cur = conn.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, name, created_at, last_used_at, "
            "request_count, plan, monthly_quota, rpm_limit, over_limit_mode, "
            "llm_base_url, llm_model, llm_api_key_enc, llm_key_hint, llm_updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _hash(raw), raw[:_KEY_PREFIX_LEN], row["name"], _now(),
                row["last_used_at"], row["request_count"], row["plan"],
                row["monthly_quota"], row["rpm_limit"],
                _norm_mode(row["over_limit_mode"]), row["llm_base_url"],
                row["llm_model"], row["llm_api_key_enc"], row["llm_key_hint"],
                row["llm_updated_at"],
            ),
        )
        new_id = cur.lastrowid
        conn.execute("UPDATE usage_daily SET key_id = ? WHERE key_id = ?", (new_id, key_id))
        conn.execute("UPDATE ocr_runs SET key_id = ? WHERE key_id = ?", (new_id, key_id))
        conn.execute("UPDATE clients SET key_id = ? WHERE key_id = ?", (new_id, key_id))
        conn.execute(
            "UPDATE api_keys SET is_active = 0, llm_api_key_enc = NULL, llm_key_hint = NULL "
            "WHERE id = ?",
            (key_id,),
        )
        new_row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (new_id,)).fetchone()
    logger.info("Rotated API key id=%s -> id=%s name=%r", key_id, new_id, row["name"])
    return raw, _row_to_info(new_row)


def verify_api_key(raw_key: str) -> Optional[APIKeyInfo]:
    """Look up a key by its hash; return its record if active, else None."""
    if not raw_key:
        return None
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (_hash(raw_key),)
        ).fetchone()
        if row is None or not row["is_active"]:
            return None
        conn.execute(
            "UPDATE api_keys SET last_used_at = ?, request_count = request_count + 1 WHERE id = ?",
            (_now(), row["id"]),
        )
    return _row_to_info(row)


def bootstrap_key() -> None:
    """Ensure DB exists and optionally seed a bootstrap key from env."""
    _init_db()
    if BOOTSTRAP_API_KEY:
        with _conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM api_keys WHERE key_hash = ?",
                (_hash(BOOTSTRAP_API_KEY),),
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO api_keys (key_hash, key_prefix, name, created_at) VALUES (?, ?, ?, ?)",
                    (
                        _hash(BOOTSTRAP_API_KEY),
                        BOOTSTRAP_API_KEY[:_KEY_PREFIX_LEN],
                        "bootstrap",
                        _now(),
                    ),
                )
                logger.info("Seeded bootstrap API key from OCR_BOOTSTRAP_API_KEY")
    seed_first_admin()


# --- Usage tracking --------------------------------------------------------
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record_usage(key_id: int) -> None:
    """Increment the per-day usage counter for a key (call after a successful
    billable request)."""
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO usage_daily (key_id, day, count) VALUES (?, ?, 1)
            ON CONFLICT(key_id, day) DO UPDATE SET count = count + 1
            """,
            (key_id, _today()),
        )


def record_ocr_run(
    key_id: int,
    *,
    filename: Optional[str] = None,
    document_type: Optional[str] = None,
    lang: Optional[str] = None,
    text_blocks: int = 0,
    tables: int = 0,
    pages: int = 0,
    chars: int = 0,
    files_count: int = 1,
    table_detection: bool = False,
    llm_parsing: bool = False,
) -> None:
    """Append one row to the per-run OCR history log for a client key.

    Called after a successful /ocr so the client portal can show a history.
    """
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO ocr_runs
                (key_id, created_at, filename, document_type, lang, text_blocks,
                 tables, pages, chars, files_count, table_detection, llm_parsing)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key_id, _now(), filename, document_type, lang, text_blocks,
                tables, pages, chars, int(max(1, files_count)),
                int(bool(table_detection)), int(bool(llm_parsing)),
            ),
        )


def get_ocr_history(key_id: int, limit: int = 50, offset: int = 0) -> list[OcrRun]:
    """Most-recent-first list of recorded OCR runs for a key."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ocr_runs WHERE key_id = ? ORDER BY created_at DESC, id DESC "
            "LIMIT ? OFFSET ?",
            (key_id, limit, offset),
        ).fetchall()
    return [
        OcrRun(
            id=r["id"],
            created_at=r["created_at"],
            filename=r["filename"],
            document_type=r["document_type"],
            lang=r["lang"],
            text_blocks=r["text_blocks"],
            tables=r["tables"],
            pages=r["pages"],
            chars=r["chars"],
            files_count=r["files_count"] if "files_count" in r.keys() else 1,
            table_detection=bool(r["table_detection"]),
            llm_parsing=bool(r["llm_parsing"]),
        )
        for r in rows
    ]


def get_api_key(key_id: int) -> Optional[APIKeyInfo]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (key_id,)
        ).fetchone()
    return _row_to_info(row) if row else None


def get_usage_summary(key_id: int) -> UsageSummary:
    today = _today()
    month = today[:7]
    with _conn() as conn:
        info = conn.execute(
            "SELECT request_count, last_used_at, created_at, monthly_quota, rpm_limit, plan, over_limit_mode "
            "FROM api_keys WHERE id = ?",
            (key_id,),
        ).fetchone()
        row = conn.execute(
            "SELECT "
            "COALESCE(SUM(count), 0) AS total, "
            "COALESCE(SUM(CASE WHEN day = ? THEN count END), 0) AS today, "
            "COALESCE(SUM(CASE WHEN day LIKE ? THEN count END), 0) AS month "
            "FROM usage_daily WHERE key_id = ?",
            (today, month + "%", key_id),
        ).fetchone()
    if info is None:
        return UsageSummary()
    try:
        created = datetime.fromisoformat(info["created_at"])
        days = max(0, (datetime.now(timezone.utc) - created).days)
    except Exception:
        days = 0
    quota = info["monthly_quota"]
    used_month = row["month"]
    remaining = (quota - used_month) if quota else None
    return UsageSummary(
        # Billable OCR requests only: api_keys.request_count also counts
        # plain authenticated calls (dashboard polling), which would inflate
        # this stat every time the portal refreshes.
        total_requests=row["total"],
        requests_today=row["today"],
        requests_this_month=used_month,
        last_used_at=info["last_used_at"],
        days_since_created=days,
        monthly_quota=quota,
        remaining=remaining,
        plan=info["plan"],
        rpm_limit=info["rpm_limit"],
        over_limit_mode=_norm_mode(info["over_limit_mode"]),
        over_quota=bool(quota) and used_month >= quota,
    )


def get_usage_series(key_id: int, days: int = 30) -> list[UsagePoint]:
    days = max(1, min(days, 365))
    with _conn() as conn:
        rows = conn.execute(
            "SELECT day, count FROM usage_daily WHERE key_id = ?", (key_id,)
        ).fetchall()
    counts = {r["day"]: r["count"] for r in rows}
    series = []
    for i in range(days - 1, -1, -1):
        d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
        series.append(UsagePoint(day=d, count=counts.get(d, 0)))
    return series


def get_admin_summary() -> AdminSummary:
    today = _today()
    with _conn() as conn:
        total, active = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(is_active), 0) FROM api_keys"
        ).fetchone()
        req_total = conn.execute(
            "SELECT COALESCE(SUM(count), 0) FROM usage_daily"
        ).fetchone()[0]
        req_today = conn.execute(
            "SELECT COALESCE(SUM(count), 0) FROM usage_daily WHERE day = ?",
            (today,),
        ).fetchone()[0]
    return AdminSummary(
        total_clients=total,
        active_clients=active or 0,
        total_requests=req_total,
        requests_today=req_today,
    )


# --- Quota & rate-limit enforcement ---------------------------------------
def _check_rate_limit(key_id: int, rpm: Optional[int]) -> None:
    """Raise 429 if the key exceeded its per-minute rate limit.

    Uses an in-process sliding window (see _rate_windows note above).
    """
    if not rpm:
        return
    now = time.monotonic()
    with _rate_lock:
        win = _rate_windows.setdefault(key_id, deque())
        cutoff = now - 60.0
        while win and win[0] < cutoff:
            win.popleft()
        if len(win) >= rpm:
            retry = int(round(60 - (now - win[0]))) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {rpm} requests/minute.",
                headers={"Retry-After": str(max(1, retry))},
            )
        win.append(now)


def enforce_limits(client: APIKeyInfo) -> dict:
    """Check monthly quota then per-minute rate limit for a billable request.

    Call BEFORE the expensive OCR work. Behaviour when the monthly quota is
    exhausted depends on the key's over_limit_mode:
      - 'hard' (default): raise HTTPException(429) so the request is blocked.
      - 'soft': keep serving, but return over_quota=True so the caller can flag
        the response (header/dashboard) instead of rejecting it.
    Rate limiting always blocks (429) regardless of mode. Returns limit info
    suitable for response headers.
    """
    quota = client.monthly_quota
    remaining: Optional[int] = None
    over_quota = False
    if quota:
        used = get_usage_summary(client.id).requests_this_month
        if used >= quota:
            over_quota = True
            if _norm_mode(client.over_limit_mode) != "soft":
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Monthly quota of {quota} requests exhausted. "
                        "It resets next month, or contact support to raise it."
                    ),
                )
        # Report what will be left AFTER this request is counted (>=0 for soft).
        remaining = max(0, quota - used - 1)
    _check_rate_limit(client.id, client.rpm_limit)
    return {
        "monthly_quota": quota,
        "remaining": remaining,
        "rpm_limit": client.rpm_limit,
        "over_quota": over_quota,
        "over_limit_mode": _norm_mode(client.over_limit_mode),
    }


# --- Human authentication (admins, client accounts, sessions) --------------
_PBKDF2_ITERS = 210_000


def _hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 with a per-password random salt (stdlib only)."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$", 1)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _PBKDF2_ITERS
        )
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# -- Admin accounts ----------------------------------------------------------
def _admin_public(row: sqlite3.Row) -> AdminPublic:
    return AdminPublic(
        id=row["id"], email=row["email"], name=row["name"],
        is_active=bool(row["is_active"]), created_at=row["created_at"],
        last_login_at=row["last_login_at"],
    )


def create_admin(email: str, password: str, name: str) -> Optional[AdminPublic]:
    email = (email or "").strip().lower()
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO admins (email, name, pw_hash, created_at) VALUES (?, ?, ?, ?)",
                (email, name.strip() or email, _hash_password(password), _now()),
            )
            row = conn.execute("SELECT * FROM admins WHERE id = ?", (cur.lastrowid,)).fetchone()
        logger.info("Created admin id=%s email=%s", row["id"], email)
        return _admin_public(row)
    except sqlite3.IntegrityError:
        return None  # duplicate email


def get_admin(admin_id: int) -> Optional[AdminPublic]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM admins WHERE id = ?", (admin_id,)).fetchone()
    return _admin_public(row) if row else None


def list_admins() -> list[AdminPublic]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM admins ORDER BY id").fetchall()
    return [_admin_public(r) for r in rows]


def active_admin_count() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM admins WHERE is_active = 1").fetchone()["c"]


def verify_admin(email: str, password: str) -> Optional[AdminPublic]:
    email = (email or "").strip().lower()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM admins WHERE email = ?", (email,)).fetchone()
    if row is None or not row["is_active"] or not _verify_password(password, row["pw_hash"]):
        return None
    return _admin_public(row)


def touch_admin_login(admin_id: int) -> None:
    with _conn() as conn:
        conn.execute("UPDATE admins SET last_login_at = ? WHERE id = ?", (_now(), admin_id))


def set_admin_active(admin_id: int, active: bool) -> bool:
    with _conn() as conn:
        cur = conn.execute("UPDATE admins SET is_active = ? WHERE id = ?", (1 if active else 0, admin_id))
        return cur.rowcount > 0


# -- Client portal accounts --------------------------------------------------
def _client_account(row: sqlite3.Row) -> ClientAccount:
    info = get_api_key(row["key_id"])
    return ClientAccount(
        id=row["id"], email=row["email"], name=row["name"], key_id=row["key_id"],
        key_prefix=info.key_prefix if info else "", is_active=bool(row["is_active"]),
        created_at=row["created_at"], last_login_at=row["last_login_at"],
    )


def create_client(email: str, password: str, name: str, key_id: int) -> Optional[ClientAccount]:
    email = (email or "").strip().lower()
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO clients (email, name, pw_hash, key_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (email, name.strip() or email, _hash_password(password), key_id, _now()),
            )
            row = conn.execute("SELECT * FROM clients WHERE id = ?", (cur.lastrowid,)).fetchone()
        logger.info("Created client account id=%s email=%s key_id=%s", row["id"], email, key_id)
        return _client_account(row)
    except sqlite3.IntegrityError:
        return None


def get_client(client_id: int) -> Optional[ClientAccount]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
    return _client_account(row) if row else None


def verify_client(email: str, password: str) -> Optional[ClientAccount]:
    email = (email or "").strip().lower()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM clients WHERE email = ?", (email,)).fetchone()
    if row is None or not row["is_active"] or not _verify_password(password, row["pw_hash"]):
        return None
    return _client_account(row)


def touch_client_login(client_id: int) -> None:
    with _conn() as conn:
        conn.execute("UPDATE clients SET last_login_at = ? WHERE id = ?", (_now(), client_id))


# -- Sessions (opaque, server-side, revocable) -------------------------------
def create_session(principal: str, principal_id: int, ttl_seconds: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, principal, principal_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (_hash(token), principal, principal_id, _now(), expires),
        )
    return token


def resolve_session(raw_token: Optional[str]) -> Optional[dict]:
    if not raw_token:
        return None
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE token_hash = ?", (_hash(raw_token),)
        ).fetchone()
    if row is None or row["revoked_at"] is not None:
        return None
    if row["expires_at"] <= _now():  # ISO strings compare lexicographically (same format/tz)
        return None
    return {"principal": row["principal"], "id": row["principal_id"]}


def revoke_session(raw_token: Optional[str]) -> None:
    if not raw_token:
        return
    with _conn() as conn:
        conn.execute(
            "UPDATE sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (_now(), _hash(raw_token)),
        )


def revoke_principal(principal: str, principal_id: int) -> None:
    """Invalidate every active session for a principal (e.g. on logout-everywhere)."""
    with _conn() as conn:
        conn.execute(
            "UPDATE sessions SET revoked_at = ? WHERE principal = ? AND principal_id = ? AND revoked_at IS NULL",
            (_now(), principal, principal_id),
        )


def seed_first_admin() -> None:
    """Create the initial admin so the portal is reachable. Runs when empty."""
    if count_admins_exists():
        return
    if ADMIN_EMAIL and ADMIN_PASSWORD:
        create_admin(ADMIN_EMAIL, ADMIN_PASSWORD, ADMIN_EMAIL.split("@")[0])
        logger.info("Seeded initial admin from OCR_ADMIN_EMAIL")
    else:
        dev_email = "admin@localhost"
        dev_pw = secrets.token_urlsafe(12)
        create_admin(dev_email, dev_pw, "Administrator")
        logger.warning(
            "No OCR_ADMIN_EMAIL/OCR_ADMIN_PASSWORD set. Created dev admin "
            "email=%s password=%s  (CHANGE IT; set env vars for production)",
            dev_email, dev_pw,
        )


def count_admins_exists() -> bool:
    with _conn() as conn:
        return conn.execute("SELECT 1 FROM admins LIMIT 1").fetchone() is not None


# --- Demo-mode account helpers --------------------------------------------
def _admin_row_by_email(email: str) -> Optional[sqlite3.Row]:
    with _conn() as conn:
        return conn.execute("SELECT * FROM admins WHERE email = ?", (email,)).fetchone()


def _client_row_by_email(email: str) -> Optional[sqlite3.Row]:
    with _conn() as conn:
        return conn.execute("SELECT * FROM clients WHERE email = ?", (email,)).fetchone()


def _reset_admin_password(admin_id: int, password: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE admins SET pw_hash = ?, is_active = 1 WHERE id = ?",
            (_hash_password(password), admin_id),
        )


def _reset_client_password(client_id: int, password: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE clients SET pw_hash = ?, is_active = 1 WHERE id = ?",
            (_hash_password(password), client_id),
        )


def ensure_demo_admin() -> AdminPublic:
    """Return the fixed demo admin, creating it if absent and resetting its
    password + active flag so the demo login always succeeds."""
    row = _admin_row_by_email(DEMO_ADMIN_EMAIL)
    if row is None:
        admin = create_admin(DEMO_ADMIN_EMAIL, DEMO_ADMIN_PASSWORD, "Demo Administrator")
        if admin is None:  # race/duplicate
            row = _admin_row_by_email(DEMO_ADMIN_EMAIL)
            admin = _admin_public(row)
        return admin
    _reset_admin_password(row["id"], DEMO_ADMIN_PASSWORD)
    return _admin_public(row)


def ensure_demo_client() -> ClientAccount:
    """Return the fixed demo client (with its own API key), creating it if
    absent and resetting its password + active flag for a reliable demo login."""
    row = _client_row_by_email(DEMO_CLIENT_EMAIL)
    if row is None:
        _raw, key_info = create_api_key(DEMO_CLIENT_NAME, plan="demo")
        account = create_client(DEMO_CLIENT_EMAIL, DEMO_CLIENT_PASSWORD, DEMO_CLIENT_NAME, key_info.id)
        if account is None:  # duplicate email created concurrently
            row = _client_row_by_email(DEMO_CLIENT_EMAIL)
            account = _client_account(row)
            revoke_api_key(key_info.id)  # don't leak the orphan key
        return account
    _reset_client_password(row["id"], DEMO_CLIENT_PASSWORD)
    return _client_account(row)


# --- FastAPI dependencies --------------------------------------------------
async def require_api_key(
    x_api_key: Optional[str] = Security(_api_key_header),
    authorization: Optional[str] = Security(_bearer_header),
) -> APIKeyInfo:
    """Reject requests without a valid, active API key."""
    raw = x_api_key
    if not raw and authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            raw = parts[1]
    if not raw:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide it in the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    record = verify_api_key(raw)
    if record is None:
        raise HTTPException(status_code=403, detail="Invalid or revoked API key.")
    return record


async def require_admin(
    x_admin_token: Optional[str] = Security(_admin_header),
) -> None:
    """Guard the key-management endpoints with the admin token."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints are disabled: OCR_ADMIN_TOKEN is not configured.",
        )
    if not x_admin_token or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token.")


# --- Session helpers & session-based dependencies --------------------------
def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    return None


def _set_session_cookie(response: Response, token: str, max_age: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE, value=token, max_age=max_age,
        httponly=True, samesite="lax", secure=COOKIE_SECURE, path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


async def require_admin_session(
    request: Request,
    x_admin_token: Optional[str] = Security(_admin_header),
    authorization: Optional[str] = Security(_bearer_header),
) -> AdminPublic:
    """Authorize an admin via session cookie/bearer, or the legacy static token."""
    token = request.cookies.get(SESSION_COOKIE) or _bearer_token(authorization)
    sess = resolve_session(token)
    if sess and sess["principal"] == "admin":
        admin = get_admin(sess["id"])
        if admin and admin.is_active:
            return admin
    # Backward-compat: a configured OCR_ADMIN_TOKEN still authorizes headless/API use.
    if ADMIN_TOKEN and x_admin_token and hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        return AdminPublic(id=0, email="token", name="token-admin", created_at=_now())
    raise HTTPException(status_code=401, detail="Admin authentication required.")


async def require_client_session(
    request: Request,
    authorization: Optional[str] = Security(_bearer_header),
) -> ClientAccount:
    """Authorize a logged-in client portal session (cookie or bearer session token)."""
    token = request.cookies.get(SESSION_COOKIE) or _bearer_token(authorization)
    sess = resolve_session(token)
    if not sess or sess["principal"] != "client":
        raise HTTPException(status_code=401, detail="Client login required.")
    account = get_client(sess["id"])
    if account is None or not account.is_active:
        raise HTTPException(status_code=403, detail="Client account is inactive.")
    return account


async def resolve_client(
    request: Request,
    x_api_key: Optional[str] = Security(_api_key_header),
    authorization: Optional[str] = Security(_bearer_header),
) -> APIKeyInfo:
    """Resolve the calling client for billable endpoints.

    Accepts EITHER a programmatic API key (X-API-Key header, or Bearer that
    matches a key) OR a logged-in client portal session (cookie / bearer session
    token). Always yields an APIKeyInfo so quota/rate/usage logic is identical.
    """
    if x_api_key:
        record = verify_api_key(x_api_key)
        if record:
            return record
        raise HTTPException(status_code=403, detail="Invalid or revoked API key.")
    bearer = _bearer_token(authorization)
    if bearer:
        record = verify_api_key(bearer)
        if record:
            return record
    token = request.cookies.get(SESSION_COOKIE) or bearer
    sess = resolve_session(token)
    if sess and sess["principal"] == "client":
        account = get_client(sess["id"])
        if account and account.is_active:
            info = get_api_key(account.key_id)
            if info and info.is_active:
                _bump_key_usage(info.id)
                return info
            raise HTTPException(status_code=403, detail="Client API key is revoked.")
    raise HTTPException(
        status_code=401,
        detail="Missing credentials. Provide an X-API-Key header or log in to the client portal.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def _bump_key_usage(key_id: int) -> None:
    """Mirror verify_api_key's bookkeeping for session-authenticated calls."""
    with _conn() as conn:
        conn.execute(
            "UPDATE api_keys SET last_used_at = ?, request_count = request_count + 1 WHERE id = ?",
            (_now(), key_id),
        )


# --- Admin router (key management) ----------------------------------------
router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin_session)])


@router.post("/keys", response_model=APIKeyCreated, summary="Issue a new API key")
async def admin_create_key(body: APIKeyCreateRequest) -> APIKeyCreated:
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required.")
    raw, info = create_api_key(
        name,
        monthly_quota=body.monthly_quota,
        rpm_limit=body.rpm_limit,
        plan=body.plan,
        over_limit_mode=body.over_limit_mode,
    )
    # The raw key is returned ONLY here; store it now, it cannot be retrieved later.
    return APIKeyCreated(key=raw, **info.model_dump())


@router.patch("/keys/{key_id}", response_model=APIKeyInfo, summary="Update a client's plan/limits")
async def admin_update_key(key_id: int, body: APIKeyUpdateRequest) -> APIKeyInfo:
    # exclude_unset -> only fields the caller actually sent are written,
    # so an explicit null clears a limit while an omitted field is left alone.
    info = update_api_key(key_id, body.model_dump(exclude_unset=True))
    if info is None:
        raise HTTPException(status_code=404, detail="Key not found.")
    return info


@router.get("/keys", response_model=APIKeyList, summary="List API keys")
async def admin_list_keys() -> APIKeyList:
    return APIKeyList(keys=list_api_keys())


@router.put("/keys/{key_id}/llm", response_model=APIKeyInfo, summary="Set a client's own LLM credentials")
async def admin_set_llm(key_id: int, body: LLMConfigUpdate) -> APIKeyInfo:
    """Store a per-client LLM endpoint/model/key used when that client asks for
    ``parse_doc``. The API key is write-only: responses only carry its last four
    characters. Omitted fields are left unchanged, null reverts to the server
    default (OCR_LLM_*), and an empty api_key clears the stored secret."""
    info = update_llm_config(key_id, **body.model_dump(exclude_unset=True))
    if info is None:
        raise HTTPException(status_code=404, detail="Key not found.")
    return info


@router.delete("/keys/{key_id}/llm", response_model=APIKeyInfo, summary="Clear a client's LLM credentials")
async def admin_clear_llm(key_id: int) -> APIKeyInfo:
    """Drop every per-client LLM override, so the client uses the server defaults."""
    info = update_llm_config(key_id, base_url=None, model=None, api_key="")
    if info is None:
        raise HTTPException(status_code=404, detail="Key not found.")
    return info


@router.get("/clients", response_model=list[ClientProfile], summary="All clients with usage")
async def admin_clients() -> list[ClientProfile]:
    """Full client list with per-client usage/quota, for the admin dashboard table."""
    result: list[ClientProfile] = []
    for info in list_api_keys():
        result.append(
            ClientProfile(
                id=info.id,
                name=info.name,
                key_prefix=info.key_prefix,
                created_at=info.created_at,
                is_active=info.is_active,
                plan=info.plan,
                monthly_quota=info.monthly_quota,
                rpm_limit=info.rpm_limit,
                over_limit_mode=info.over_limit_mode,
                llm_base_url=info.llm_base_url,
                llm_model=info.llm_model,
                llm_key_hint=info.llm_key_hint,
                llm_updated_at=info.llm_updated_at,
                usage=get_usage_summary(info.id),
            )
        )
    return result


@router.post("/keys/{key_id}/rotate", response_model=APIKeyCreated, summary="Rotate a client's API key")
async def admin_rotate_key(key_id: int) -> APIKeyCreated:
    """Issue a replacement key and revoke the old one (use when a key is lost).

    Only hashes are stored, so a lost key cannot be displayed again; rotating is
    the supported way to hand a client a working credential. Plan, quota, rate
    limit and the client's LLM settings carry over, usage and OCR history stay
    with the client, and the old key stops working immediately. The new key is
    returned only in this response.
    """
    result = rotate_api_key(key_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Key not found or already revoked.")
    raw, info = result
    return APIKeyCreated(key=raw, **info.model_dump())


@router.delete("/keys/{key_id}", response_model=APIKeyInfo, summary="Revoke an API key")
async def admin_revoke_key(key_id: int) -> APIKeyInfo:
    if not revoke_api_key(key_id):
        raise HTTPException(status_code=404, detail="Key not found or already revoked.")
    for info in list_api_keys():
        if info.id == key_id:
            return info
    raise HTTPException(status_code=404, detail="Key not found.")


@router.get("/summary", response_model=AdminSummary, summary="Fleet-wide dashboard stats")
async def admin_summary() -> AdminSummary:
    return get_admin_summary()


@router.get("/keys/{key_id}/usage", response_model=UsageReport, summary="Per-client usage series")
async def admin_key_usage(key_id: int, days: int = 30) -> UsageReport:
    if get_api_key(key_id) is None:
        raise HTTPException(status_code=404, detail="Key not found.")
    return UsageReport(key_id=key_id, series=get_usage_series(key_id, days))


# --- Client self-service router -------------------------------------------
# A client may authenticate with their API key OR a logged-in portal session.
client_router = APIRouter(prefix="/client", tags=["Client Portal"])


@client_router.get("/me", response_model=ClientProfile, summary="My account & usage")
async def client_me(client: APIKeyInfo = Depends(resolve_client)) -> ClientProfile:
    return ClientProfile(
        id=client.id,
        name=client.name,
        key_prefix=client.key_prefix,
        created_at=client.created_at,
        is_active=client.is_active,
        plan=client.plan,
        monthly_quota=client.monthly_quota,
        rpm_limit=client.rpm_limit,
        over_limit_mode=client.over_limit_mode,
        usage=get_usage_summary(client.id),
    )


@client_router.get("/me/usage", response_model=UsageReport, summary="My usage series")
async def client_usage(
    client: APIKeyInfo = Depends(resolve_client),
    days: int = 30,
) -> UsageReport:
    return UsageReport(key_id=client.id, series=get_usage_series(client.id, days))


@client_router.get("/me/history", response_model=OcrHistory, summary="My recent OCR runs")
async def client_history(
    client: APIKeyInfo = Depends(resolve_client),
    limit: int = 50,
    offset: int = 0,
) -> OcrHistory:
    """Paginated, most-recent-first log of this client's OCR runs."""
    return OcrHistory(key_id=client.id, runs=get_ocr_history(client.id, limit, offset))


# --- Admin account management (admins can add/remove other admins) ---------
@router.get("/admins", response_model=list[AdminPublic], summary="List admin accounts")
async def admin_list_admins() -> list[AdminPublic]:
    return list_admins()


@router.post("/admins", response_model=AdminPublic, summary="Create an admin account")
async def admin_create_admin(body: SignupRequest) -> AdminPublic:
    admin = create_admin(body.email, body.password, body.name)
    if admin is None:
        raise HTTPException(status_code=409, detail="An admin with that email already exists.")
    return admin


@router.delete("/admins/{admin_id}", response_model=AdminPublic, summary="Deactivate an admin account")
async def admin_deactivate_admin(admin_id: int) -> AdminPublic:
    target = get_admin(admin_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Admin not found.")
    if target.is_active and active_admin_count() <= 1:
        raise HTTPException(status_code=400, detail="Cannot deactivate the last active admin.")
    set_admin_active(admin_id, False)
    revoke_principal("admin", admin_id)
    return get_admin(admin_id)  # type: ignore[return-value]


# --- Public authentication router -----------------------------------------
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

# Simple in-process throttle for public signups to blunt scripted mass self-signup.
_signup_times: deque = deque()
_SIGNUP_WINDOW_SECONDS = 60
_SIGNUP_MAX_PER_WINDOW = int(os.getenv("OCR_SIGNUP_RATE_PER_MIN", "10"))


def _throttle_signup() -> None:
    now = time.monotonic()
    with _rate_lock:
        while _signup_times and _signup_times[0] < now - _SIGNUP_WINDOW_SECONDS:
            _signup_times.popleft()
        if len(_signup_times) >= _SIGNUP_MAX_PER_WINDOW:
            raise HTTPException(
                status_code=429,
                detail="Too many sign-up attempts right now. Please try again in a minute.",
            )
        _signup_times.append(now)


def _email_ok(email: str) -> bool:
    e = (email or "").strip()
    return "@" in e and "." in e.split("@")[-1] and len(e) <= 200


@auth_router.post("/signup", response_model=SignupResponse, summary="Create a client account (public)")
async def signup(body: SignupRequest, response: Response) -> SignupResponse:
    """Public self-signup: creates a portal login AND issues an API key."""
    _throttle_signup()
    email = body.email.strip().lower()
    if not _email_ok(email):
        raise HTTPException(status_code=400, detail="A valid email is required.")
    raw_key, key_info = create_api_key(body.name)
    account = create_client(email, body.password, body.name, key_info.id)
    if account is None:
        revoke_api_key(key_info.id)  # don't leak an orphan key
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    token = create_session("client", account.id, CLIENT_SESSION_TTL_SECONDS)
    _set_session_cookie(response, token, CLIENT_SESSION_TTL_SECONDS)
    return SignupResponse(
        account=account,
        api_key=raw_key,
        message="Account created. Save the API key now — it is shown only once.",
    )


@auth_router.post("/login", response_model=AuthUser, summary="Unified portal login (role auto-detected)")
async def unified_login(body: LoginRequest, response: Response) -> AuthUser:
    """Single sign-in for all principals: tries admin, then client.

    The response ``type`` ('admin' | 'client') tells the portal which dashboard
    to render. Actual authorization stays server-side: /admin/* requires an
    admin session and /client/* requires a client session, so a client cannot
    reach admin data even if the front end is tampered with.
    """
    admin = verify_admin(body.email, body.password)
    if admin is not None:
        touch_admin_login(admin.id)
        token = create_session("admin", admin.id, ADMIN_SESSION_TTL_SECONDS)
        _set_session_cookie(response, token, ADMIN_SESSION_TTL_SECONDS)
        return AuthUser(type="admin", id=admin.id, email=admin.email, name=admin.name, is_admin=True)
    account = verify_client(body.email, body.password)
    if account is not None:
        touch_client_login(account.id)
        token = create_session("client", account.id, CLIENT_SESSION_TTL_SECONDS)
        _set_session_cookie(response, token, CLIENT_SESSION_TTL_SECONDS)
        return AuthUser(type="client", id=account.id, email=account.email, name=account.name)
    raise HTTPException(status_code=401, detail="Invalid email or password.")


@auth_router.post("/demo", response_model=AuthUser, summary="One-click demo login (OCR_DEMO_MODE only)")
async def demo_login(body: DemoLoginRequest, response: Response) -> AuthUser:
    """Seed/refresh a fixed demo account and sign it in immediately.

    Guarded by ``OCR_DEMO_MODE``; returns 404 when demo mode is off so the
    route is effectively invisible in production. Sets the same session cookie
    as the normal login, so downstream role enforcement is unchanged.
    """
    if not DEMO_MODE:
        raise HTTPException(status_code=404, detail="Demo mode is disabled.")
    role = (body.role or "client").strip().lower()
    if role == "admin":
        admin = ensure_demo_admin()
        touch_admin_login(admin.id)
        token = create_session("admin", admin.id, ADMIN_SESSION_TTL_SECONDS)
        _set_session_cookie(response, token, ADMIN_SESSION_TTL_SECONDS)
        return AuthUser(type="admin", id=admin.id, email=admin.email, name=admin.name, is_admin=True)
    account = ensure_demo_client()
    touch_client_login(account.id)
    token = create_session("client", account.id, CLIENT_SESSION_TTL_SECONDS)
    _set_session_cookie(response, token, CLIENT_SESSION_TTL_SECONDS)
    return AuthUser(type="client", id=account.id, email=account.email, name=account.name)


@auth_router.post("/client/login", response_model=AuthUser, summary="Client portal login")
async def client_login(body: LoginRequest, response: Response) -> AuthUser:
    account = verify_client(body.email, body.password)
    if account is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    touch_client_login(account.id)
    token = create_session("client", account.id, CLIENT_SESSION_TTL_SECONDS)
    _set_session_cookie(response, token, CLIENT_SESSION_TTL_SECONDS)
    return AuthUser(type="client", id=account.id, email=account.email, name=account.name)


@auth_router.post("/admin/login", response_model=AuthUser, summary="Admin portal login")
async def admin_login(body: LoginRequest, response: Response) -> AuthUser:
    admin = verify_admin(body.email, body.password)
    if admin is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    touch_admin_login(admin.id)
    token = create_session("admin", admin.id, ADMIN_SESSION_TTL_SECONDS)
    _set_session_cookie(response, token, ADMIN_SESSION_TTL_SECONDS)
    return AuthUser(type="admin", id=admin.id, email=admin.email, name=admin.name, is_admin=True)


@auth_router.post("/logout", summary="Log out (revokes this session)")
async def logout(request: Request, response: Response) -> dict:
    revoke_session(request.cookies.get(SESSION_COOKIE))
    _clear_session_cookie(response)
    return {"ok": True}


@auth_router.get("/me", response_model=AuthUser, summary="Who am I (session probe)")
async def auth_me(
    request: Request,
    authorization: Optional[str] = Security(_bearer_header),
) -> AuthUser:
    token = request.cookies.get(SESSION_COOKIE) or _bearer_token(authorization)
    sess = resolve_session(token)
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if sess["principal"] == "admin":
        admin = get_admin(sess["id"])
        if admin and admin.is_active:
            return AuthUser(type="admin", id=admin.id, email=admin.email, name=admin.name, is_admin=True)
    else:
        account = get_client(sess["id"])
        if account and account.is_active:
            return AuthUser(type="client", id=account.id, email=account.email, name=account.name)
    raise HTTPException(status_code=401, detail="Not authenticated.")
