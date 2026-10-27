"""Pydantic schemas for request/response models."""

from typing import Optional
from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    """OCR request parameters."""
    lang: str = Field(default="en", description="Language code (en, ch, japan, korean, etc.)")
    detect_table: bool = Field(default=False, description="Enable table detection")
    parse_document: bool = Field(default=False, description="Enable structured document parsing via LLM")


class TextBlock(BaseModel):
    """A single detected text block with position info."""
    text: str
    confidence: float
    bbox: list[list[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


class TableCell(BaseModel):
    """A single cell in a detected table."""
    row: int
    col: int
    text: str
    bbox: Optional[list[list[float]]] = None


class Table(BaseModel):
    """A detected table with cells."""
    cells: list[TableCell]
    bbox: Optional[list[list[float]]] = None


class PageResult(BaseModel):
    """Extraction results for a single page (PDF page or workbook sheet)."""
    page: int
    text_blocks: list[TextBlock] = []
    tables: list[Table] = []


class DetectedDocument(BaseModel):
    """What the LLM decided the upload is (only with ``auto_detect`` parsing)."""
    document_type: str = Field(default="unknown", description="Canonical type: passport, identity_card, driving_license, receipt, not_a_document, ...")
    label: Optional[str] = Field(default=None, description="Human-readable title, e.g. 'Cafe receipt'")
    category: Optional[str] = Field(default=None, description="identity | financial | correspondence | reference | not_a_document | other")
    is_document: bool = Field(default=True, description="False when the image is not a document at all (a photo, packaging, ...)")
    confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Classifier confidence, 0-1")
    summary: Optional[str] = Field(default=None, description="One-sentence description of the content")


class OCRResponse(BaseModel):
    """OCR response with extracted data."""
    text_blocks: list[TextBlock] = []
    full_text: str = ""
    tables: list[Table] = []
    document: Optional[dict] = None
    detected: Optional[DetectedDocument] = None
    language: str = "en"
    document_type: str = "image"  # file format: image, pdf, docx, xlsx
    pages: list[PageResult] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.1.0"


class APIKeyCreateRequest(BaseModel):
    """Request to issue a new API key."""
    name: str = Field(..., description="Client/label for this key", min_length=1, max_length=120)
    plan: Optional[str] = Field(default=None, description="Plan label, e.g. free/pro")
    monthly_quota: Optional[int] = Field(default=None, ge=1, description="Max OCR requests per calendar month; omit = unlimited")
    rpm_limit: Optional[int] = Field(default=None, ge=1, description="Max requests per minute; omit = unlimited")
    over_limit_mode: Optional[str] = Field(default=None, description="When monthly quota is hit: 'hard' = HTTP 429 (default), 'soft' = keep serving but flag it")


class APIKeyUpdateRequest(BaseModel):
    """Partial update of a client's plan/limits/status."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    plan: Optional[str] = None
    monthly_quota: Optional[int] = Field(default=None, ge=1)
    rpm_limit: Optional[int] = Field(default=None, ge=1)
    is_active: Optional[bool] = None
    over_limit_mode: Optional[str] = Field(default=None, description="'hard' | 'soft'")


class LLMConfigUpdate(BaseModel):
    """Admin-supplied LLM credentials for one client key.

    Omitted fields are left unchanged. ``base_url``/``model`` set to null revert
    to the server default (OCR_LLM_*), and an empty ``api_key`` clears the
    stored secret. The key itself is never returned by any endpoint.
    """
    base_url: Optional[str] = Field(
        default=None, max_length=500,
        description="OpenAI-compatible base URL, e.g. https://openrouter.ai/api/v1. Null = server default.",
    )
    model: Optional[str] = Field(
        default=None, max_length=120,
        description="Model id for this client. Null = server default.",
    )
    api_key: Optional[str] = Field(
        default=None, max_length=400,
        description="Provider API key (stored encrypted, shown only as a 4-char hint). Empty string clears it.",
    )


class APIKeyInfo(BaseModel):
    """Public (non-secret) representation of an API key."""
    id: int
    name: str
    key_prefix: str
    created_at: str
    last_used_at: Optional[str] = None
    request_count: int = 0
    is_active: bool = True
    plan: Optional[str] = None
    monthly_quota: Optional[int] = None
    rpm_limit: Optional[int] = None
    over_limit_mode: str = "hard"
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_key_hint: Optional[str] = None
    llm_updated_at: Optional[str] = None


class APIKeyCreated(APIKeyInfo):
    """API key info plus the raw secret, returned only once at creation."""
    key: str


class APIKeyList(BaseModel):
    """List of API keys."""
    keys: list[APIKeyInfo] = []


class UsagePoint(BaseModel):
    """A single day's request count."""
    day: str            # YYYY-MM-DD (UTC)
    count: int


class UsageSummary(BaseModel):
    """Rolling usage stats for one client key."""
    total_requests: int = 0
    requests_today: int = 0
    requests_this_month: int = 0
    last_used_at: Optional[str] = None
    days_since_created: int = 0
    monthly_quota: Optional[int] = None
    remaining: Optional[int] = None
    plan: Optional[str] = None
    rpm_limit: Optional[int] = None
    over_limit_mode: str = "hard"
    over_quota: bool = False


class ClientProfile(BaseModel):
    """Self-service profile returned to a logged-in client."""
    id: int
    name: str
    key_prefix: str
    created_at: str
    is_active: bool = True
    plan: Optional[str] = None
    monthly_quota: Optional[int] = None
    rpm_limit: Optional[int] = None
    over_limit_mode: str = "hard"
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_key_hint: Optional[str] = None
    llm_updated_at: Optional[str] = None
    usage: UsageSummary


class UsageReport(BaseModel):
    """Time-series usage for a key."""
    key_id: int
    series: list[UsagePoint] = []


class OcrRun(BaseModel):
    """A single recorded OCR run, for the client history log."""
    id: int
    created_at: str
    filename: Optional[str] = None
    document_type: Optional[str] = None
    lang: Optional[str] = None
    text_blocks: int = 0
    tables: int = 0
    pages: int = 0
    chars: int = 0
    files_count: int = 1
    table_detection: bool = False
    llm_parsing: bool = False


class OcrHistory(BaseModel):
    """Recent OCR runs for a client key."""
    key_id: int
    runs: list[OcrRun] = []


class AdminSummary(BaseModel):
    """Fleet-wide stats for the admin portal dashboard."""
    total_clients: int = 0
    active_clients: int = 0
    total_requests: int = 0
    requests_today: int = 0


# ---------------------------------------------------------------------------
# Human authentication (admin & client portal login / self-signup)
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    """Email + password credential pair for portal login."""
    email: str = Field(..., description="Account email", max_length=200)
    password: str = Field(..., description="Account password", min_length=1, max_length=200)


class SignupRequest(BaseModel):
    """Public client self-signup: creates a portal account + an API key."""
    email: str = Field(..., description="Email used to log in", max_length=200)
    password: str = Field(..., description="At least 8 characters", min_length=8, max_length=200)
    name: str = Field(..., description="Company / contact name", min_length=1, max_length=120)


class DemoLoginRequest(BaseModel):
    """Demo shortcut login (only active when OCR_DEMO_MODE=true)."""
    role: str = Field(default="client", description="'admin' or 'client'")


class AuthUser(BaseModel):
    """The currently authenticated principal for a session."""
    type: str = Field(..., description="'admin' or 'client'")
    id: int
    email: str
    name: str
    is_admin: bool = False


class AdminPublic(BaseModel):
    """Non-secret representation of an admin account."""
    id: int
    email: str
    name: str
    is_active: bool = True
    created_at: str
    last_login_at: Optional[str] = None


class ClientAccount(BaseModel):
    """A client portal account and the API key it owns."""
    id: int
    email: str
    name: str
    key_id: int
    key_prefix: str
    is_active: bool = True
    created_at: str
    last_login_at: Optional[str] = None


class SignupResponse(BaseModel):
    """Returned once at signup: the account plus the raw API key."""
    account: ClientAccount
    api_key: str = Field(..., description="Raw API key for programmatic /ocr calls. Shown only once.")
    message: str = ""
