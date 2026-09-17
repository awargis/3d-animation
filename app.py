"""
AI Cinematic STEM Animator — Hardened Multi-Provider Edition
==============================================================

Pipeline
--------
1. Ingest text / image / PDF.
2. Extract the question (including scanned/image-heavy PDFs).
3. Optional mathematical/content verification.
4. Build a Visual Bible for cross-scene consistency.
5. Generate a beat-based cinematic storyboard.
6. Generate one Manim scene at a time.
7. Validate generated Python with AST + compile checks.
8. Render with Manim inside a controlled subprocess.
9. Inspect the actual MP4 with ffprobe.
10. Self-heal recoverable failures.
11. Normalize scene videos before assembly.
12. Concatenate with optional xfade transitions.
13. Apply optional cinematic grade and music.
14. Display the final MP4 and diagnostics.

IMPORTANT SECURITY NOTE
-----------------------
Generated Python is untrusted code. AST validation below is only a
best-effort guardrail. It is NOT a security boundary.

For a public deployment, run Manim in a separate sandbox/container with:
- no network access,
- non-root user,
- CPU/memory limits,
- restricted filesystem,
- execution timeout,
- temporary working directory.

Local installation
------------------
pip install -r requirements.txt

System requirements
-------------------
Manim Community Edition
FFmpeg + ffprobe
LaTeX is NOT required by this app because generated scenes use Text()
rather than MathTex/Tex.

Run
---
streamlit run app.py
"""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import py_compile
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st
from PIL import Image


# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(
    page_title="AI Cinematic STEM Animator",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 AI Cinematic STEM Animator")
st.caption(
    "Question → verified understanding → visual bible → storyboard → "
    "self-healing Manim scenes → cinematic final cut"
)


# ============================================================================
# Constants
# ============================================================================

APP_VERSION = "2.0.0"

DEFAULT_MAX_PDF_PAGES = 6
DEFAULT_PDF_DPI = 180
DEFAULT_RENDER_TIMEOUT = 600

ALLOWED_GENERATED_IMPORTS = {"manim", "numpy"}
BLOCKED_MODULE_PREFIXES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "pathlib",
    "shutil",
    "glob",
    "pickle",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
    "importlib",
    "builtins",
    "resource",
}
BLOCKED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "system",
    "popen",
    "spawn",
    "run",
    "call",
    "check_call",
    "check_output",
}
BLOCKED_ATTRIBUTES = {
    "__globals__",
    "__builtins__",
    "__import__",
    "__subclasses__",
    "__bases__",
    "__mro__",
}

THEME_PALETTES = {
    "Deep Space (navy/cyan glow)": {
        "bg": "#03060f",
        "primary": "#38f2ff",
        "accent": "#ff5da2",
        "secondary": "#7d8cff",
        "text": "#f5f9ff",
    },
    "Blueprint (dark slate/amber)": {
        "bg": "#0b1420",
        "primary": "#ffb454",
        "accent": "#5ac8ff",
        "secondary": "#7ee2c5",
        "text": "#eef3f8",
    },
    "Aurora (violet/teal gradient)": {
        "bg": "#0a0518",
        "primary": "#7ee8fa",
        "accent": "#c17cff",
        "secondary": "#75ffd2",
        "text": "#f7f2ff",
    },
}

GEMINI_MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
]

GROQ_MODEL_OPTIONS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

GROQ_VISION_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

QUALITY_OPTIONS = {
    "Low (-ql, fast draft)": "-ql",
    "Medium (-qm)": "-qm",
    "High (-qh, 1080p60)": "-qh",
}


# ============================================================================
# Exceptions and data classes
# ============================================================================

class ProviderError(Exception):
    """An LLM provider failed and the router may try another provider."""


class ValidationError(Exception):
    """Generated content failed a structural or safety validation."""


class RenderError(Exception):
    """A Manim/FFmpeg operation failed."""


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    model: str
    supports_vision: bool
    priority: int


@dataclass
class SceneResult:
    index: int
    title: str
    success: bool = False
    video_path: Optional[str] = None
    actual_duration: float = 0.0
    code: str = ""
    log: list[str] = field(default_factory=list)
    attempts: int = 0
    failure_reason: Optional[str] = None


# ============================================================================
# General helpers
# ============================================================================

def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def now_label() -> str:
    return time.strftime("%H:%M:%S")


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def check_system_dependencies() -> list[str]:
    missing = []
    for executable in ("ffmpeg", "ffprobe", "manim"):
        if not command_exists(executable):
            missing.append(executable)
    return missing


def normalize_json_text(raw_text: str) -> str:
    """Extract the most likely JSON object/array from an LLM response."""
    text = safe_text(raw_text)
    text = re.sub(r"^\s*```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    # Remove common preambles while preserving the JSON payload.
    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [x for x in (first_obj, first_arr) if x >= 0]
    if starts:
        start = min(starts)
        text = text[start:]

    # Try to trim after the final JSON delimiter.
    last_obj = text.rfind("}")
    last_arr = text.rfind("]")
    end = max(last_obj, last_arr)
    if end >= 0:
        text = text[: end + 1]

    return text.strip()


def parse_json_object(raw_text: str, label: str) -> dict[str, Any]:
    cleaned = normalize_json_text(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"{label} returned invalid JSON: {exc.msg} at character {exc.pos}."
        ) from exc

    if not isinstance(data, dict):
        raise ValidationError(f"{label} must return a JSON object.")
    return data


def sanitize_code(raw_text: str) -> str:
    """Remove Markdown fences and obvious prose around generated Python."""
    text = safe_text(raw_text)
    text = re.sub(r"^\s*```(?:python|py|Python)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    # If an LLM accidentally adds prose before imports, start at from manim.
    match = re.search(r"(?m)^\s*from\s+manim\s+import\s+\*", text)
    if match:
        text = text[match.start():]

    return text.strip()


# ============================================================================
# Provider layer
# ============================================================================

def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "rate limit",
            "rate_limit",
            "resource_exhausted",
            "quota",
            "too many requests",
        )
    )


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def call_gemini(
    api_key: str,
    model: str,
    system_prompt: Optional[str],
    user_parts: list[Any],
) -> str:
    try:
        from google import genai
    except ImportError as exc:
        raise ProviderError(
            "Gemini SDK is not installed. Run: pip install google-genai"
        ) from exc

    try:
        client = genai.Client(api_key=api_key)

        # Gemini accepts text and PIL image objects directly.
        contents: list[Any] = []
        if system_prompt:
            contents.append(system_prompt)
        contents.extend(user_parts)

        response = client.models.generate_content(
            model=model,
            contents=contents,
        )

        text = getattr(response, "text", None)
        if not text:
            raise ProviderError("Gemini returned an empty response.")
        return text
    except ProviderError:
        raise
    except Exception as exc:
        prefix = "Gemini rate/quota error" if is_rate_limit_error(exc) else "Gemini error"
        raise ProviderError(f"{prefix}: {exc}") from exc


def build_chat_messages(
    system_prompt: Optional[str],
    user_parts: list[Any],
    supports_vision: bool,
) -> list[dict[str, Any]]:
    text_chunks: list[str] = []
    image_blocks: list[dict[str, Any]] = []

    for part in user_parts:
        if isinstance(part, Image.Image):
            if not supports_vision:
                raise ProviderError(
                    "This provider/model does not support vision, "
                    "but this call requires an image."
                )
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_data_url(part)},
                }
            )
        else:
            text_chunks.append(str(part))

    joined_text = "\n\n".join(text_chunks)
    user_content: Any

    if image_blocks:
        user_content = image_blocks
        if joined_text:
            user_content.append({"type": "text", "text": joined_text})
    else:
        user_content = joined_text

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def call_groq(
    api_key: str,
    model: str,
    system_prompt: Optional[str],
    user_parts: list[Any],
    supports_vision: bool,
) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise ProviderError(
            "Groq SDK is not installed. Run: pip install groq"
        ) from exc

    messages = build_chat_messages(system_prompt, user_parts, supports_vision)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        text = response.choices[0].message.content
        if not text:
            raise ProviderError("Groq returned an empty response.")
        return text
    except ProviderError:
        raise
    except Exception as exc:
        prefix = "Groq rate/quota error" if is_rate_limit_error(exc) else "Groq error"
        raise ProviderError(f"{prefix}: {exc}") from exc


def call_openai(
    api_key: str,
    model: str,
    system_prompt: Optional[str],
    user_parts: list[Any],
    supports_vision: bool,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProviderError(
            "OpenAI SDK is not installed. Run: pip install openai"
        ) from exc

    messages = build_chat_messages(system_prompt, user_parts, supports_vision)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        text = response.choices[0].message.content
        if not text:
            raise ProviderError("OpenAI returned an empty response.")
        return text
    except ProviderError:
        raise
    except Exception as exc:
        prefix = "OpenAI rate/quota error" if is_rate_limit_error(exc) else "OpenAI error"
        raise ProviderError(f"{prefix}: {exc}") from exc


class LLMRouter:
    """Provider router with priority ordering and vision-aware fallback."""

    def __init__(self, providers: list[ProviderConfig]):
        if not providers:
            raise ValueError("At least one provider is required.")
        self.providers = sorted(providers, key=lambda p: p.priority)

    def generate(
        self,
        system_prompt: Optional[str],
        user_parts: list[Any],
        need_vision: bool = False,
        log: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, str]:
        attempts: list[str] = []

        for provider in self.providers:
            if need_vision and not provider.supports_vision:
                attempts.append(
                    f"{provider.name}: skipped because this call needs vision."
                )
                continue

            try:
                if provider.name == "Gemini":
                    text = call_gemini(
                        provider.api_key,
                        provider.model,
                        system_prompt,
                        user_parts,
                    )
                elif provider.name == "Groq":
                    text = call_groq(
                        provider.api_key,
                        provider.model,
                        system_prompt,
                        user_parts,
                        provider.supports_vision,
                    )
                elif provider.name == "OpenAI":
                    text = call_openai(
                        provider.api_key,
                        provider.model,
                        system_prompt,
                        user_parts,
                        provider.supports_vision,
                    )
                else:
                    raise ProviderError(f"Unknown provider: {provider.name}")

                if log:
                    log(
                        f"✓ {now_label()} — served by "
                        f"**{provider.name} / {provider.model}**"
                    )
                return text, provider.name

            except ProviderError as exc:
                attempts.append(f"{provider.name}: {exc}")
                if log:
                    log(
                        f"⚠️ {now_label()} — {provider.name} unavailable: "
                        f"{str(exc)[:400]}"
                    )

        raise RuntimeError(
            "All configured AI providers failed for this request.\n\n"
            + "\n".join(attempts)
        )


# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("🔑 AI Providers")
st.sidebar.caption(
    "Configure one provider or several. Lower priority numbers are tried first."
)

with st.sidebar.expander("Gemini", expanded=True):
    gemini_key = st.text_input(
        "API Key",
        type="password",
        key="gemini_key",
    )
    gemini_model = st.selectbox(
        "Model",
        GEMINI_MODEL_OPTIONS,
        key="gemini_model",
    )
    gemini_priority = st.number_input(
        "Priority",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
        key="gemini_priority",
    )

with st.sidebar.expander("Groq"):
    groq_key = st.text_input(
        "API Key",
        type="password",
        key="groq_key",
    )
    groq_model = st.selectbox(
        "Model",
        GROQ_MODEL_OPTIONS,
        key="groq_model",
        help="Scout is the vision-capable option in this list.",
    )
    groq_priority = st.number_input(
        "Priority",
        min_value=1,
        max_value=3,
        value=2,
        step=1,
        key="groq_priority",
    )

with st.sidebar.expander("OpenAI"):
    openai_key = st.text_input(
        "API Key",
        type="password",
        key="openai_key",
    )
    openai_model = st.text_input(
        "Model ID",
        value="gpt-4o-mini",
        key="openai_model",
    )
    openai_vision = st.checkbox(
        "Model supports vision",
        value=True,
        key="openai_vision",
    )
    openai_priority = st.number_input(
        "Priority",
        min_value=1,
        max_value=3,
        value=3,
        step=1,
        key="openai_priority",
    )

providers: list[ProviderConfig] = []

if gemini_key.strip():
    providers.append(
        ProviderConfig(
            "Gemini",
            gemini_key.strip(),
            gemini_model,
            True,
            int(gemini_priority),
        )
    )

if groq_key.strip():
    providers.append(
        ProviderConfig(
            "Groq",
            groq_key.strip(),
            groq_model,
            groq_model in GROQ_VISION_MODELS,
            int(groq_priority),
        )
    )

if openai_key.strip():
    providers.append(
        ProviderConfig(
            "OpenAI",
            openai_key.strip(),
            openai_model.strip(),
            bool(openai_vision),
            int(openai_priority),
        )
    )

if providers:
    order = " → ".join(
        f"{p.name} ({p.priority})"
        for p in sorted(providers, key=lambda x: x.priority)
    )
    st.sidebar.success(f"Fallback order: {order}")
else:
    st.sidebar.warning("No provider configured.")

st.sidebar.divider()
st.sidebar.header("⚙️ Render Configuration")

quality_label = st.sidebar.selectbox(
    "Render Quality",
    list(QUALITY_OPTIONS.keys()),
)
quality_flag = QUALITY_OPTIONS[quality_label]

target_duration = st.sidebar.slider(
    "Target total duration (seconds)",
    min_value=10,
    max_value=120,
    value=45,
    step=5,
)

style_theme = st.sidebar.selectbox(
    "Visual theme",
    list(THEME_PALETTES.keys()),
)
palette = THEME_PALETTES[style_theme]

use_crossfade = st.sidebar.checkbox(
    "Cinematic crossfade transitions",
    value=True,
)
transition_len = (
    st.sidebar.slider(
        "Crossfade length (s)",
        min_value=0.2,
        max_value=1.5,
        value=0.6,
        step=0.1,
    )
    if use_crossfade
    else 0.0
)

use_cinematic_grade = st.sidebar.checkbox(
    "Cinematic color grade + vignette + letterbox",
    value=True,
)

max_heal_attempts = st.sidebar.slider(
    "Max self-heal attempts per scene",
    min_value=1,
    max_value=4,
    value=3,
)

strict_duration = st.sidebar.checkbox(
    "Strict scene duration validation",
    value=False,
    help=(
        "If enabled, scenes must be much closer to their target duration. "
        "Recommended for final exports; disable for faster experimentation."
    ),
)

verify_content = st.sidebar.checkbox(
    "AI mathematical/content verification",
    value=True,
    help=(
        "Adds a verification pass before animation generation. "
        "Recommended for JEE/NEET educational content."
    ),
)

music_file = st.sidebar.file_uploader(
    "Optional background music (MP3)",
    type=["mp3"],
)
music_volume = (
    st.sidebar.slider(
        "Music volume",
        min_value=0.0,
        max_value=1.0,
        value=0.12,
    )
    if music_file
    else 0.0
)


# ============================================================================
# Input
# ============================================================================

input_type = st.radio(
    "Input Type",
    ["Text Question", "Image / Screenshot", "PDF"],
    horizontal=True,
)

question_text = ""
question_image: Optional[Image.Image] = None
pdf_bytes: Optional[bytes] = None

if input_type == "Text Question":
    question_text = st.text_area(
        "Question Text",
        height=120,
        placeholder=(
            "Example: A charged particle enters a uniform magnetic field "
            "with velocity at an angle to the field. Visualize its trajectory "
            "and explain the motion."
        ),
    )

elif input_type == "Image / Screenshot":
    uploaded_img = st.file_uploader(
        "Upload Question Image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_img:
        try:
            question_image = Image.open(uploaded_img).convert("RGB")
            st.image(question_image, caption="Uploaded Question", width=500)
        except Exception as exc:
            st.error(f"Could not open image: {exc}")

else:
    uploaded_pdf = st.file_uploader(
        "Upload Question PDF",
        type=["pdf"],
    )
    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.read()
        st.success(
            f"Loaded PDF — {len(pdf_bytes) / 1024:.0f} KB"
        )


# ============================================================================
# PDF ingestion
# ============================================================================

def extract_pdf_pages(
    raw_bytes: bytes,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
    dpi: int = DEFAULT_PDF_DPI,
) -> list[tuple[str, Optional[Image.Image]]]:
    """
    Extract text and optionally render PDF pages.

    For STEM PDFs, visual information can be important even when text
    extraction succeeds, so pages with short/ambiguous extraction are
    rasterized. We also retain extracted text.
    """
    if not raw_bytes:
        raise ValueError("PDF is empty.")

    text_pages: list[tuple[str, Optional[Image.Image]]] = []

    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages[:max_pages]:
                text = (page.extract_text() or "").strip()
                text_pages.append((text, None))
    except Exception:
        text_pages = []

    # Render pages as visual context. This is intentionally more conservative
    # than only rasterizing when ALL text extraction fails.
    try:
        import fitz

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        output: list[tuple[str, Optional[Image.Image]]] = []

        for index in range(min(len(doc), max_pages)):
            extracted_text = (
                text_pages[index][0]
                if index < len(text_pages)
                else ""
            )

            page = doc[index]
            pix = page.get_pixmap(
                dpi=dpi,
                alpha=False,
            )
            img = Image.open(
                io.BytesIO(pix.tobytes("png"))
            ).convert("RGB")

            # Send visual pages to the model when text is short or when the
            # page contains likely diagrams/equations. Keeping the image also
            # makes scanned PDFs work.
            output.append((extracted_text, img))

        doc.close()
        return output

    except Exception as exc:
        # If rasterization is unavailable, return text-only extraction.
        if text_pages:
            return text_pages
        raise ValueError(
            "Could not extract text or rasterize this PDF. "
            f"Install PyMuPDF/pdfplumber. Details: {exc}"
        ) from exc


EXTRACTION_SYSTEM_PROMPT = """
You are a meticulous STEM question reader.

Extract the question without changing its mathematical or physical meaning.

Return:
1. Exact/reconstructed problem statement.
2. All given values, symbols, constraints and units.
3. What is being asked.
4. Core concepts.
5. Important diagrams/visual objects visible in the input.
6. Ambiguities or unreadable portions.

Do not invent missing values.
If something is unclear, explicitly say so.
"""


def build_question_context(
    router: LLMRouter,
    log: Callable[[str], None],
) -> str:
    if question_text.strip():
        return (
            "QUESTION SOURCE: typed text\n\n"
            + question_text.strip()
        )

    if question_image is not None:
        extracted, provider = router.generate(
            EXTRACTION_SYSTEM_PROMPT,
            [question_image],
            need_vision=True,
            log=log,
        )
        return (
            f"QUESTION SOURCE: image, interpreted by {provider}\n\n"
            + extracted.strip()
        )

    if pdf_bytes is not None:
        pages = extract_pdf_pages(pdf_bytes)

        text_chunks: list[str] = []
        image_parts: list[Image.Image] = []

        for i, (text, img) in enumerate(pages, start=1):
            if text:
                text_chunks.append(
                    f"PDF PAGE {i} TEXT:\n{text}"
                )

            if img is not None:
                image_parts.append(img)

        parts: list[Any] = []
        if text_chunks:
            parts.append(
                "Extracted PDF text:\n\n"
                + "\n\n---\n\n".join(text_chunks)
            )

        # Limit visual pages to avoid sending a huge PDF to every provider.
        parts.extend(image_parts[:DEFAULT_MAX_PDF_PAGES])

        if not parts:
            raise ValueError("Could not extract any usable PDF content.")

        extracted, provider = router.generate(
            EXTRACTION_SYSTEM_PROMPT,
            parts,
            need_vision=bool(image_parts),
            log=log,
        )

        return (
            f"QUESTION SOURCE: PDF, interpreted by {provider}\n\n"
            + extracted.strip()
        )

    raise ValueError("No question input was provided.")


# ============================================================================
# Mathematical/content verification
# ============================================================================

VERIFICATION_SYSTEM_PROMPT = """
You are the mathematical/physics content verifier for an educational
animation system used for competitive-exam preparation.

Analyze the supplied question and produce ONLY valid JSON:

{
  "problem_summary": "...",
  "given_data": ["..."],
  "asked": "...",
  "concepts": ["..."],
  "correct_method": ["step 1", "step 2", "..."],
  "key_equations": ["..."],
  "final_result": "...",
  "common_misconceptions": ["..."],
  "visual_truths": [
    "A statement that must be visually represented correctly."
  ],
  "verification_notes": [
    "Any ambiguity or caveat."
  ]
}

Rules:
- Do not invent data.
- Check signs, units, definitions and mathematical relationships.
- If a numerical answer cannot be determined, say so.
- Distinguish exact facts from assumptions.
- The visual_truths list is especially important: these statements will
  later constrain the animation.
"""


def verify_question(
    router: LLMRouter,
    context: str,
    log: Callable[[str], None],
) -> tuple[dict[str, Any], str]:
    raw, provider = router.generate(
        VERIFICATION_SYSTEM_PROMPT,
        [context],
        need_vision=False,
        log=log,
    )
    data = parse_json_object(raw, "Verification")

    required = [
        "problem_summary",
        "given_data",
        "asked",
        "concepts",
        "correct_method",
        "key_equations",
        "final_result",
        "common_misconceptions",
        "visual_truths",
        "verification_notes",
    ]

    for key in required:
        if key not in data:
            raise ValidationError(
                f"Verification JSON is missing required field '{key}'."
            )

    return data, provider


# ============================================================================
# Visual Bible
# ============================================================================

VISUAL_BIBLE_SYSTEM_PROMPT = """
You are the visual director for a premium STEM animation system.

Create a compact VISUAL BIBLE that every scene must obey.

Return ONLY valid JSON:

{
  "coordinate_system": {
    "type": "3D/cartesian/number-line/diagram/etc",
    "x_range": [-6, 6],
    "y_range": [-4, 4],
    "z_range": [-4, 4],
    "camera_phi": 70,
    "camera_theta": -45,
    "camera_zoom": 0.9
  },
  "objects": [
    {
      "id": "unique_object_id",
      "description": "...",
      "role": "main/supporting/reference",
      "color_role": "primary/accent/secondary/text"
    }
  ],
  "visual_motifs": ["..."],
  "equation_style": "...",
  "animation_language": [
    "How important concepts should enter/move/change."
  ],
  "continuity_rules": [
    "Exact rules that must remain unchanged across scenes."
  ],
  "avoid": [
    "Visual choices that would create conceptual confusion."
  ]
}

Rules:
- Optimize for conceptual clarity, not decoration.
- Preserve mathematical/physical meaning.
- Use the supplied palette.
- Do not invent physical objects that are irrelevant.
"""


def generate_visual_bible(
    router: LLMRouter,
    context: str,
    verification: Optional[dict[str, Any]],
    palette: dict[str, str],
    log: Callable[[str], None],
) -> tuple[dict[str, Any], str]:
    verification_text = (
        json.dumps(verification, indent=2)
        if verification
        else "No separate verification pass was requested."
    )

    user_prompt = f"""
QUESTION CONTEXT:
{context}

VERIFICATION:
{verification_text}

FIXED PALETTE:
{json.dumps(palette, indent=2)}
"""

    raw, provider = router.generate(
        VISUAL_BIBLE_SYSTEM_PROMPT,
        [user_prompt],
        need_vision=False,
        log=log,
    )

    bible = parse_json_object(raw, "Visual Bible")

    for key in (
        "coordinate_system",
        "objects",
        "visual_motifs",
        "equation_style",
        "animation_language",
        "continuity_rules",
        "avoid",
    ):
        if key not in bible:
            raise ValidationError(
                f"Visual Bible is missing required field '{key}'."
            )

    return bible, provider


# ============================================================================
# Storyboard
# ============================================================================

STORYBOARD_SYSTEM_PROMPT = """
You are a world-class STEM educational director.

Create a short cinematic storyboard that teaches the supplied question.

Return ONLY valid JSON:

{
  "scenes": [
    {
      "title": "string",
      "purpose": "What the student learns in this scene.",
      "description": "Detailed visual direction.",
      "teaching_strategy": "Why this scene exists pedagogically.",
      "beats": [
        {
          "action": "specific visual action",
          "explanation": "what the student should understand",
          "duration_seconds": 2.0
        }
      ],
      "duration_seconds": 8.0
    }
  ]
}

Rules:
- Total scene durations should be within 10% of the requested total.
- Each scene: 5–15 seconds.
- Prefer 4–8 meaningful scenes rather than many tiny scenes.
- Use multiple visual beats instead of a giant wait.
- Mathematical correctness always beats visual spectacle.
- Do not introduce an object that contradicts the Visual Bible.
- Maintain the same coordinate ranges, object IDs, color roles and camera
  language across scenes.
- Use subject-appropriate pedagogy.

For probability: expose sample space, conditioning, probability mass and
misconceptions visually.

For coordinate geometry: establish points/axes/vectors before equations.

For calculus: show changing quantity, geometric meaning and then symbolic form.

For algebra/binomial: show structure/pattern before the general formula.

For physics: show physical cause → effect, directions, reference frame and
conservation/constraints where relevant.

Do not force a generic hook if it damages conceptual clarity.
"""


def validate_storyboard(
    data: dict[str, Any],
    target_duration: float,
) -> list[dict[str, Any]]:
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValidationError("Storyboard contains no scenes.")

    validated: list[dict[str, Any]] = []

    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValidationError(f"Scene {index} is not an object.")

        for key in (
            "title",
            "purpose",
            "description",
            "teaching_strategy",
            "beats",
            "duration_seconds",
        ):
            if key not in scene:
                raise ValidationError(
                    f"Scene {index} is missing '{key}'."
                )

        try:
            duration = float(scene["duration_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Scene {index} has invalid duration."
            ) from exc

        if not 5 <= duration <= 15:
            raise ValidationError(
                f"Scene {index} duration must be between 5 and 15 seconds."
            )

        beats = scene["beats"]
        if not isinstance(beats, list) or not beats:
            raise ValidationError(
                f"Scene {index} must contain at least one beat."
            )

        for beat_index, beat in enumerate(beats, start=1):
            if not isinstance(beat, dict):
                raise ValidationError(
                    f"Scene {index}, beat {beat_index} is invalid."
                )
            if "action" not in beat or "duration_seconds" not in beat:
                raise ValidationError(
                    f"Scene {index}, beat {beat_index} is incomplete."
                )

        clean = dict(scene)
        clean["duration_seconds"] = duration
        validated.append(clean)

    total = sum(s["duration_seconds"] for s in validated)
    tolerance = max(3.0, target_duration * 0.10)

    # Do not reject a useful storyboard just because the model rounded
    # durations. Instead rescale durations proportionally.
    if total > 0 and abs(total - target_duration) > tolerance:
        scale = target_duration / total
        for scene in validated:
            scene["duration_seconds"] = clamp(
                scene["duration_seconds"] * scale,
                5.0,
                15.0,
            )

    return validated


def generate_storyboard(
    router: LLMRouter,
    context: str,
    verification: Optional[dict[str, Any]],
    visual_bible: dict[str, Any],
    target_duration: int,
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], str]:
    user_prompt = f"""
TARGET TOTAL DURATION: {target_duration} seconds

QUESTION:
{context}

VERIFICATION:
{json.dumps(verification, indent=2) if verification else "Not available."}

VISUAL BIBLE:
{json.dumps(visual_bible, indent=2)}

PALETTE:
{json.dumps(palette, indent=2)}
"""

    raw, provider = router.generate(
        STORYBOARD_SYSTEM_PROMPT,
        [user_prompt],
        need_vision=False,
        log=log,
    )

    data = parse_json_object(raw, "Storyboard")
    scenes = validate_storyboard(data, target_duration)

    return scenes, provider


# ============================================================================
# Generated Python validation
# ============================================================================

def module_name_allowed(name: str) -> bool:
    root = name.split(".")[0]
    return root in ALLOWED_GENERATED_IMPORTS


def validate_generated_python(code: str) -> None:
    """
    Best-effort AST security and structural validation.

    This DOES NOT make generated Python safe for hostile/public execution.
    Full isolation requires an OS/container sandbox.
    """
    code = sanitize_code(code)

    if len(code) > 100_000:
        raise ValidationError("Generated scene code is unreasonably large.")

    if "MathTex(" in code or "Tex(" in code:
        raise ValidationError(
            "Generated code uses MathTex/Tex. This app requires Text()."
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValidationError(
            f"Generated Python has a syntax error: {exc.msg} "
            f"at line {exc.lineno}."
        ) from exc

    class_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]

    if class_names != ["GeneratedScene"]:
        raise ValidationError(
            "Generated code must define exactly one class named GeneratedScene."
        )

    generated_scene = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "GeneratedScene"
    )

    base_names = []
    for base in generated_scene.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)

    if "ThreeDScene" not in base_names:
        raise ValidationError(
            "GeneratedScene must inherit from ThreeDScene."
        )

    # Imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not module_name_allowed(alias.name):
                    raise ValidationError(
                        f"Unsafe/unsupported import blocked: {alias.name}"
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module is None or not module_name_allowed(node.module):
                raise ValidationError(
                    f"Unsafe/unsupported import blocked: {node.module}"
                )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_CALLS:
                    raise ValidationError(
                        f"Blocked function call detected: {node.func.id}()"
                    )

            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_CALLS:
                    raise ValidationError(
                        f"Blocked method call detected: .{node.func.attr}()"
                    )

        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRIBUTES:
                raise ValidationError(
                    f"Blocked attribute detected: .{node.attr}"
                )

    # Reject obvious attempts to use dunder machinery.
    if re.search(r"__\w+__", code):
        raise ValidationError(
            "Dunder-based introspection is blocked in generated scene code."
        )

    # Educational rendering should not need these.
    for forbidden in (
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "importlib",
        "ctypes",
        "pickle",
        "eval(",
        "exec(",
    ):
        if forbidden in code:
            raise ValidationError(
                f"Blocked unsafe construct detected: {forbidden}"
            )

    if "from manim import *" not in code:
        raise ValidationError(
            "Generated code must begin with 'from manim import *'."
        )


def compile_check(code: str, tmpdir: str) -> Optional[str]:
    path = os.path.join(tmpdir, "syntax_check.py")

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)

        py_compile.compile(
            path,
            doraise=True,
        )
        return None
    except py_compile.PyCompileError as exc:
        return str(exc)


# ============================================================================
# Scene code generation
# ============================================================================

SCENE_SYSTEM_PROMPT = """
You are an expert Manim Community Edition animator creating ONE scene in a
multi-scene educational STEM film.

OUTPUT:
Return ONLY runnable Python code.
No Markdown fences.
No explanations.

HARD REQUIREMENTS:
1. Start with:
   from manim import *
   import numpy as np

2. Define exactly:
   class GeneratedScene(ThreeDScene):

3. Do NOT use MathTex or Tex. Use Text().

4. Do NOT import anything other than manim and numpy.

5. Do NOT use os, sys, subprocess, socket, requests, urllib, eval, exec,
   open, pathlib, shutil or any filesystem/network/process APIs.

6. Every self.play(...) MUST have explicit run_time.

7. Use self.wait() between meaningful beats.

8. The rendered scene should contain genuine staged teaching actions, not
   decorative motion.

9. Use the exact Visual Bible object IDs and visual language where practical.

10. Mathematical/physical relationships must be represented correctly.

11. Avoid excessive text. Show the idea first, then concise labels/equations.

12. Use a sparse background particle/star field only when it does not compete
    with the lesson.

13. Use camera motion purposefully. At least one meaningful camera movement
    is preferred, but never sacrifice readability.

14. Prefer smooth easing.

15. Keep all objects inside a readable frame.

16. If an equation is needed, use Text("...") with plain Unicode/ASCII
    notation rather than LaTeX.

DURATION:
The target scene duration is {target_duration:.1f} seconds.
Design multiple visual beats whose durations sum close to the target.
Do NOT use one giant wait as padding.

PALETTE:
{palette}

VISUAL BIBLE:
{visual_bible}

SCENE:
{scene}

VERIFIED CONTENT:
{verification}

CONTINUITY:
{continuity_note}
"""


def generate_scene_code(
    router: LLMRouter,
    scene: dict[str, Any],
    visual_bible: dict[str, Any],
    verification: Optional[dict[str, Any]],
    palette: dict[str, str],
    continuity_note: str,
    log: Callable[[str], None],
) -> tuple[str, str]:
    prompt = SCENE_SYSTEM_PROMPT.format(
        target_duration=float(scene["duration_seconds"]),
        palette=json.dumps(palette, indent=2),
        visual_bible=json.dumps(visual_bible, indent=2),
        scene=json.dumps(scene, indent=2),
        verification=(
            json.dumps(verification, indent=2)
            if verification
            else "No verification object available."
        ),
        continuity_note=continuity_note,
    )

    raw, provider = router.generate(
        None,
        [prompt],
        need_vision=False,
        log=log,
    )

    code = sanitize_code(raw)
    validate_generated_python(code)
    return code, provider


def estimate_static_duration(code: str) -> float:
    """
    Early warning only. Rendered ffprobe duration remains authoritative.
    """
    total = 0.0

    for match in re.finditer(
        r"run_time\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        code,
    ):
        total += float(match.group(1))

    for match in re.finditer(
        r"self\.wait\(\s*([0-9]+(?:\.[0-9]+)?)?\s*\)",
        code,
    ):
        total += float(match.group(1) or 1.0)

    return total


# ============================================================================
# Subprocess wrapper
# ============================================================================

def run_command(
    command: list[str],
    *,
    timeout: int,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    """
    Centralized subprocess execution.

    Generated Manim code should ideally be executed inside an external
    sandbox/container for public deployments.
    """
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(
            f"Command timed out after {timeout}s: {' '.join(command[:5])}"
        ) from exc
    except FileNotFoundError as exc:
        raise RenderError(
            f"Required executable was not found: {command[0]}"
        ) from exc
    except OSError as exc:
        raise RenderError(
            f"Could not execute command: {exc}"
        ) from exc


def run_manim_render(
    script_path: str,
    tmpdir: str,
    quality_flag: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        "manim",
        quality_flag,
        "--media_dir",
        tmpdir,
        "--custom_folders",
        "--fps",
        "30",
        script_path,
        "GeneratedScene",
    ]

    return run_command(
        command,
        timeout=DEFAULT_RENDER_TIMEOUT,
    )


# ============================================================================
# Media helpers
# ============================================================================

def find_mp4_files(root: str) -> list[str]:
    candidates: list[str] = []

    for directory, _, files in os.walk(root):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                path = os.path.join(directory, filename)
                try:
                    if os.path.getsize(path) > 1024:
                        candidates.append(path)
                except OSError:
                    continue

    return sorted(
        candidates,
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )


def find_best_mp4(root: str) -> Optional[str]:
    candidates = find_mp4_files(root)
    return candidates[0] if candidates else None


def get_video_duration(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0

    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrapper=1:nokey=1",
            path,
        ],
        timeout=30,
    )

    if result.returncode != 0:
        raise RenderError(
            "ffprobe could not inspect the video:\n"
            + result.stderr[-1500:]
        )

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RenderError(
            f"ffprobe returned an invalid duration: {result.stdout!r}"
        ) from exc

    if duration <= 0:
        raise RenderError("Video duration is zero or negative.")

    return duration


def verify_video_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise RenderError("Video file does not exist.")

    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name",
            "-of",
            "json",
            path,
        ],
        timeout=30,
    )

    if result.returncode != 0:
        raise RenderError(
            "Video validation failed:\n"
            + result.stderr[-1500:]
        )

    try:
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        return {
            "duration": float(fmt.get("duration", 0)),
            "size": int(fmt.get("size", 0)),
            "format": fmt.get("format_name", ""),
        }
    except Exception as exc:
        raise RenderError(
            "Could not parse ffprobe video metadata."
        ) from exc


# ============================================================================
# Scene rendering + self healing
# ============================================================================

def duration_is_acceptable(
    actual: float,
    target: float,
    strict: bool,
) -> bool:
    if strict:
        return target * 0.85 <= actual <= target * 1.20
    return actual >= target * 0.65


def render_scene_with_healing(
    router: LLMRouter,
    scene: dict[str, Any],
    idx: int,
    visual_bible: dict[str, Any],
    verification: Optional[dict[str, Any]],
    palette: dict[str, str],
    continuity_note: str,
    quality_flag: str,
    max_attempts: int,
    strict_duration_mode: bool,
    status_box: Any,
) -> SceneResult:
    result = SceneResult(
        index=idx,
        title=safe_text(scene.get("title")) or f"Scene {idx + 1}",
    )

    target = float(scene["duration_seconds"])
    code = ""
    provider = ""
    error_feedback: Optional[str] = None

    log = lambda message: status_box.write(message)

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        status_box.write(
            f"### Scene {idx + 1} — attempt {attempt}/{max_attempts}"
        )

        try:
            if not code:
                code, provider = generate_scene_code(
                    router,
                    scene,
                    visual_bible,
                    verification,
                    palette,
                    continuity_note,
                    log,
                )
            elif error_feedback:
                repair_prompt = f"""
Repair the following Manim scene.

SCENE:
{json.dumps(scene, indent=2)}

VISUAL BIBLE:
{json.dumps(visual_bible, indent=2)}

VERIFIED CONTENT:
{json.dumps(verification, indent=2) if verification else "None"}

PREVIOUS CODE:
{code}

ERROR / QA FEEDBACK:
{error_feedback}

Rules:
- Preserve the intended educational visual.
- Return ONLY complete corrected Python.
- Keep exactly class GeneratedScene(ThreeDScene).
- Only import manim and numpy.
- No filesystem, network, subprocess or introspection.
- Do not use MathTex/Tex.
- Every self.play must have explicit run_time.
- Target duration is {target:.1f}s.
"""

                raw, provider = router.generate(
                    None,
                    [repair_prompt],
                    need_vision=False,
                    log=log,
                )
                code = sanitize_code(raw)
                validate_generated_python(code)

            # Safety validation before touching Manim.
            validate_generated_python(code)

            static_estimate = estimate_static_duration(code)
            if static_estimate < target * 0.55:
                error_feedback = (
                    f"Static duration estimate is only "
                    f"{static_estimate:.1f}s for a {target:.1f}s target. "
                    "Add meaningful staged beats, not one giant wait."
                )
                result.log.append(
                    f"attempt {attempt}: static duration too short"
                )
                continue

            with tempfile.TemporaryDirectory(
                prefix=f"scene_{idx}_"
            ) as scene_tmp:
                compile_error = compile_check(code, scene_tmp)

                if compile_error:
                    error_feedback = (
                        "Python compile error:\n"
                        + compile_error[-4000:]
                    )
                    result.log.append(
                        f"attempt {attempt}: compile error"
                    )
                    continue

                script_path = os.path.join(
                    scene_tmp,
                    f"scene_{idx}.py",
                )

                with open(
                    script_path,
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write(code)

                try:
                    process = run_manim_render(
                        script_path,
                        scene_tmp,
                        quality_flag,
                    )
                except RenderError as exc:
                    error_feedback = str(exc)
                    result.log.append(
                        f"attempt {attempt}: render execution error"
                    )
                    continue

                if process.returncode != 0:
                    combined_error = (
                        process.stderr[-6000:]
                        + "\n"
                        + process.stdout[-2000:]
                    )
                    error_feedback = (
                        "Manim render failed:\n"
                        + combined_error
                    )
                    result.log.append(
                        f"attempt {attempt}: Manim failed"
                    )
                    continue

                mp4 = find_best_mp4(scene_tmp)

                if not mp4:
                    error_feedback = (
                        "Manim reported success but no valid MP4 "
                        "was found in the render directory."
                    )
                    result.log.append(
                        f"attempt {attempt}: no MP4"
                    )
                    continue

                try:
                    media_info = verify_video_file(mp4)
                    actual_duration = float(
                        media_info["duration"]
                    )
                except RenderError as exc:
                    error_feedback = str(exc)
                    result.log.append(
                        f"attempt {attempt}: media validation failed"
                    )
                    continue

                if not duration_is_acceptable(
                    actual_duration,
                    target,
                    strict_duration_mode,
                ):
                    error_feedback = (
                        f"Rendered video is {actual_duration:.1f}s; "
                        f"target is {target:.1f}s. "
                        "Add or extend distinct visual teaching beats. "
                        "Do not use one giant wait."
                    )
                    result.log.append(
                        f"attempt {attempt}: duration {actual_duration:.1f}s"
                    )
                    continue

                persistent_dir = tempfile.mkdtemp(
                    prefix="cinematic_scene_"
                )
                final_path = os.path.join(
                    persistent_dir,
                    f"scene_{idx}.mp4",
                )

                shutil.copy2(mp4, final_path)

                result.success = True
                result.video_path = final_path
                result.actual_duration = actual_duration
                result.code = code

                status_box.write(
                    f"✅ Scene {idx + 1} rendered — "
                    f"{actual_duration:.1f}s — {provider}"
                )
                return result

        except ValidationError as exc:
            error_feedback = str(exc)
            result.log.append(
                f"attempt {attempt}: validation error"
            )

        except Exception as exc:
            error_feedback = (
                f"Unexpected scene error: {type(exc).__name__}: {exc}"
            )
            result.log.append(
                f"attempt {attempt}: unexpected error"
            )

    result.code = code
    result.failure_reason = error_feedback

    status_box.write(
        f"⚠️ Scene {idx + 1} failed after {max_attempts} attempts."
    )
    return result


# ============================================================================
# Video normalization and assembly
# ============================================================================

def normalize_video(
    input_path: str,
    output_path: str,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """
    Normalize all scene videos to a common format before xfade/concat.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={fps},format=yuv420p"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        output_path,
    ]

    result = run_command(
        command,
        timeout=180,
    )

    if result.returncode != 0:
        raise RenderError(
            "FFmpeg normalization failed:\n"
            + result.stderr[-4000:]
        )

    if not os.path.isfile(output_path):
        raise RenderError(
            "FFmpeg reported success but normalized video was not created."
        )

    return output_path


def concat_simple(
    video_paths: list[str],
    tmpdir: str,
) -> str:
    if not video_paths:
        raise RenderError("No videos supplied for concatenation.")

    if len(video_paths) == 1:
        return video_paths[0]

    list_path = os.path.join(
        tmpdir,
        "concat_list.txt",
    )

    with open(
        list_path,
        "w",
        encoding="utf-8",
    ) as handle:
        for path in video_paths:
            escaped = path.replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    output = os.path.join(
        tmpdir,
        "concat_simple.mp4",
    )

    result = run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        timeout=300,
    )

    if result.returncode != 0:
        raise RenderError(
            "Simple concatenation failed:\n"
            + result.stderr[-4000:]
        )

    return output


def concat_with_crossfade(
    video_paths: list[str],
    durations: list[float],
    transition: float,
    tmpdir: str,
) -> str:
    if len(video_paths) <= 1:
        return video_paths[0]

    transition = max(0.05, transition)

    inputs: list[str] = []
    for path in video_paths:
        inputs.extend(["-i", path])

    filter_parts: list[str] = []

    prev_label = "[0:v]"
    running_duration = durations[0]

    for index in range(1, len(video_paths)):
        effective_transition = min(
            transition,
            max(0.05, durations[index - 1] - 0.05),
            max(0.05, durations[index] - 0.05),
        )

        offset = max(
            0.05,
            running_duration - effective_transition,
        )

        out_label = f"[xf{index}]"

        filter_parts.append(
            f"{prev_label}[{index}:v]"
            f"xfade=transition=fade:"
            f"duration={effective_transition:.3f}:"
            f"offset={offset:.3f}"
            f"{out_label}"
        )

        running_duration = (
            running_duration
            + durations[index]
            - effective_transition
        )
        prev_label = out_label

    output = os.path.join(
        tmpdir,
        "concat_crossfade.mp4",
    )

    result = run_command(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            prev_label,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        timeout=400,
    )

    if result.returncode != 0:
        raise RenderError(
            "Crossfade assembly failed:\n"
            + result.stderr[-5000:]
        )

    return output


def apply_cinematic_grade(
    input_path: str,
    tmpdir: str,
) -> str:
    output = os.path.join(
        tmpdir,
        "graded.mp4",
    )

    vf = (
        "eq=contrast=1.08:saturation=1.12:brightness=0.01,"
        "vignette=PI/4,"
        "pad=iw:round(ih*1.14):0:(oh-ih)/2:black"
    )

    result = run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        timeout=300,
    )

    if result.returncode != 0:
        raise RenderError(
            "Cinematic grade failed:\n"
            + result.stderr[-4000:]
        )

    return output


def mux_music(
    video_path: str,
    music_bytes: bytes,
    volume: float,
    tmpdir: str,
) -> str:
    music_path = os.path.join(
        tmpdir,
        "background_music.mp3",
    )

    with open(music_path, "wb") as handle:
        handle.write(music_bytes)

    output = os.path.join(
        tmpdir,
        "final_with_music.mp4",
    )

    filter_complex = (
        f"[1:a]volume={clamp(volume, 0.0, 1.0):.3f},"
        "afade=t=in:st=0:d=1,"
        "afade=t=out:st=9999:d=1[a]"
    )

    result = run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-stream_loop",
            "-1",
            "-i",
            music_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-shortest",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            output,
        ],
        timeout=300,
    )

    if result.returncode != 0:
        raise RenderError(
            "Music muxing failed:\n"
            + result.stderr[-4000:]
        )

    return output


# ============================================================================
# Main pipeline
# ============================================================================

def render_dependency_warning() -> None:
    missing = check_system_dependencies()

    if missing:
        st.warning(
            "Missing system dependency: "
            + ", ".join(missing)
            + ". The app can still display its UI, but rendering will fail "
              "until the dependency is installed."
        )


render_dependency_warning()


if st.button(
    "🚀 Generate Cinematic Animation",
    use_container_width=True,
    type="primary",
):
    if not providers:
        st.error(
            "Configure at least one AI provider in the sidebar."
        )
        st.stop()

    if (
        not question_text.strip()
        and question_image is None
        and pdf_bytes is None
    ):
        st.error(
            "Provide a text question, image, or PDF."
        )
        st.stop()

    router = LLMRouter(providers)
    output_root = tempfile.mkdtemp(
        prefix="cinematic_output_"
    )

    try:
        # --------------------------------------------------------------------
        # Stage 1 — Question understanding
        # --------------------------------------------------------------------
        with st.status(
            "🧠 Understanding the question…",
            expanded=True,
        ) as status:
            context = build_question_context(
                router,
                log=lambda m: status.write(m),
            )

            st.write(
                context[:1200]
                + ("…" if len(context) > 1200 else "")
            )

            # ---------------------------------------------------------------
            # Stage 2 — Verification
            # ---------------------------------------------------------------
            verification = None
            verification_provider = None

            if verify_content:
                status.update(
                    label="🔬 Verifying mathematical/content correctness…"
                )

                verification, verification_provider = verify_question(
                    router,
                    context,
                    log=lambda m: status.write(m),
                )

                with st.expander(
                    "🔬 Verification report",
                    expanded=False,
                ):
                    st.json(verification)

            # ---------------------------------------------------------------
            # Stage 3 — Visual Bible
            # ---------------------------------------------------------------
            status.update(
                label="🎨 Building Visual Bible…"
            )

            visual_bible, bible_provider = generate_visual_bible(
                router,
                context,
                verification,
                palette,
                log=lambda m: status.write(m),
            )

            with st.expander(
                "🎨 Visual Bible",
                expanded=False,
            ):
                st.json(visual_bible)

            # ---------------------------------------------------------------
            # Stage 4 — Storyboard
            # ---------------------------------------------------------------
            status.update(
                label="🎬 Writing teaching storyboard…"
            )

            scenes, storyboard_provider = generate_storyboard(
                router,
                context,
                verification,
                visual_bible,
                target_duration,
                log=lambda m: status.write(m),
            )

            status.update(
                label=(
                    f"📋 Storyboard ready — {len(scenes)} scenes"
                ),
                state="running",
            )

            total_storyboard_duration = sum(
                float(scene["duration_seconds"])
                for scene in scenes
            )

            st.info(
                f"Storyboard: {len(scenes)} scenes • "
                f"planned duration: {total_storyboard_duration:.1f}s • "
                f"target: {target_duration}s"
            )

            for index, scene in enumerate(scenes, start=1):
                st.write(
                    f"**Scene {index}: {scene['title']}** — "
                    f"{scene['duration_seconds']:.1f}s"
                )

        # --------------------------------------------------------------------
        # Stage 5 — Render scenes
        # --------------------------------------------------------------------
        scene_results: list[SceneResult] = []

        continuity_note = (
            "This is the first scene. Establish the Visual Bible clearly. "
            "Use the exact coordinate system and object IDs."
        )

        for index, scene in enumerate(scenes):
            box = st.status(
                f"🎞️ Rendering scene {index + 1}/{len(scenes)} — "
                f"{scene['title']}",
                expanded=True,
            )

            result = render_scene_with_healing(
                router=router,
                scene=scene,
                idx=index,
                visual_bible=visual_bible,
                verification=verification,
                palette=palette,
                continuity_note=continuity_note,
                quality_flag=quality_flag,
                max_attempts=max_heal_attempts,
                strict_duration_mode=strict_duration,
                status_box=box,
            )

            scene_results.append(result)

            if result.success:
                continuity_note = (
                    f"Continue directly after scene {index + 1} "
                    f"('{scene['title']}'). Preserve all established "
                    f"Visual Bible IDs, axes, colors, object meanings and "
                    f"camera language. Do not reset the conceptual world."
                )
                box.update(
                    label=(
                        f"✅ Scene {index + 1} complete — "
                        f"{result.actual_duration:.1f}s"
                    ),
                    state="complete",
                )
            else:
                box.update(
                    label=(
                        f"⚠️ Scene {index + 1} failed — "
                        "see diagnostics"
                    ),
                    state="error",
                )

        # --------------------------------------------------------------------
        # Stage 6 — Assemble
        # --------------------------------------------------------------------
        usable = [
            result
            for result in scene_results
            if result.success and result.video_path
        ]

        if not usable:
            st.error(
                "No scene rendered successfully."
            )

            for result in scene_results:
                with st.expander(
                    f"Scene {result.index + 1} — {result.title}"
                ):
                    st.write(result.log)
                    if result.failure_reason:
                        st.error(result.failure_reason)
                    if result.code:
                        st.code(
                            result.code,
                            language="python",
                        )

            st.stop()

        with st.status(
            "🎞️ Assembling final cut…",
            expanded=True,
        ) as status:
            raw_paths = [
                result.video_path
                for result in usable
                if result.video_path
            ]

            status.write(
                f"Normalizing {len(raw_paths)} scene video(s)…"
            )

            normalized_paths: list[str] = []
            normalized_durations: list[float] = []

            # Manim -qh is 1080p; -qm/-ql can be smaller. Normalize everything
            # before xfade so stream properties are identical.
            for index, path in enumerate(raw_paths):
                normalized = os.path.join(
                    output_root,
                    f"normalized_{index}.mp4",
                )

                normalize_video(
                    path,
                    normalized,
                    fps=30,
                    width=1920,
                    height=1080,
                )

                normalized_paths.append(normalized)
                normalized_durations.append(
                    get_video_duration(normalized)
                )

            if len(normalized_paths) == 1:
                combined = normalized_paths[0]
            elif use_crossfade:
                try:
                    status.write("Applying cinematic crossfades…")
                    combined = concat_with_crossfade(
                        normalized_paths,
                        normalized_durations,
                        transition_len,
                        output_root,
                    )
                except RenderError as exc:
                    st.warning(
                        "Crossfade failed. Falling back to simple concatenation.\n\n"
                        + str(exc)
                    )
                    combined = concat_simple(
                        normalized_paths,
                        output_root,
                    )
            else:
                combined = concat_simple(
                    normalized_paths,
                    output_root,
                )

            if use_cinematic_grade:
                status.update(
                    label="🎨 Applying cinematic grade…"
                )
                combined = apply_cinematic_grade(
                    combined,
                    output_root,
                )

            if music_file is not None:
                status.update(
                    label="🎵 Mixing background music…"
                )
                combined = mux_music(
                    combined,
                    music_file.getvalue(),
                    music_volume,
                    output_root,
                )

            # Final validation.
            final_info = verify_video_file(combined)
            total_actual = float(
                final_info["duration"]
            )

            status.update(
                label="✅ Final video validated",
                state="complete",
            )

        # --------------------------------------------------------------------
        # Stage 7 — Results
        # --------------------------------------------------------------------
        successful_count = len(usable)

        st.success(
            f"🎉 Rendered {successful_count}/{len(scenes)} scenes — "
            f"final duration {total_actual:.1f}s"
        )

        st.video(combined)

        with open(
            combined,
            "rb",
        ) as handle:
            st.download_button(
                "⬇️ Download MP4",
                handle.read(),
                file_name="cinematic_stem_animation.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        if successful_count < len(scenes):
            st.warning(
                f"{len(scenes) - successful_count} scene(s) failed. "
                "The final video contains only successfully rendered scenes."
            )

        # --------------------------------------------------------------------
        # Diagnostics / audit trail
        # --------------------------------------------------------------------
        with st.expander(
            "📊 Pipeline audit",
            expanded=False,
        ):
            st.write(
                {
                    "app_version": APP_VERSION,
                    "input_type": input_type,
                    "provider_order": [
                        f"{p.name}/{p.model}"
                        for p in sorted(
                            providers,
                            key=lambda x: x.priority,
                        )
                    ],
                    "verification_provider": verification_provider,
                    "visual_bible_provider": bible_provider,
                    "storyboard_provider": storyboard_provider,
                    "planned_duration": target_duration,
                    "actual_duration": round(total_actual, 2),
                    "successful_scenes": successful_count,
                    "total_scenes": len(scenes),
                    "quality": quality_label,
                    "theme": style_theme,
                }
            )

        for result in scene_results:
            state = "ok" if result.success else "failed"

            with st.expander(
                f"Scene {result.index + 1} — "
                f"{result.title} [{state}]"
            ):
                st.write(
                    {
                        "attempts": result.attempts,
                        "actual_duration": result.actual_duration,
                        "success": result.success,
                    }
                )

                if result.log:
                    st.write(result.log)

                if result.failure_reason:
                    st.error(result.failure_reason)

                if result.code:
                    st.code(
                        result.code,
                        language="python",
                    )

    except Exception as exc:
        st.error(
            f"❌ Pipeline failed at the application level: "
            f"{type(exc).__name__}: {exc}"
        )

        with st.expander(
            "🔧 Technical diagnostic",
            expanded=True,
        ):
            st.exception(exc)

        st.info(
            "Tip: If this happened during rendering, first check that "
            "Manim, FFmpeg and ffprobe are installed and available on PATH. "
            "If it happened during AI generation, check the provider key, "
            "model ID and provider fallback order."
        )
