"""
AI Cinematic STEM Animator — Streamlit app
============================================
Pipeline:
  1. Ingest question (text / image / PDF, incl. scanned PDFs via page-rasterization)
  2. Gemini writes a STORYBOARD (list of scenes with description + target duration)
  3. Each scene's Manim code is generated INDIVIDUALLY (small, focused context ->
     far fewer hallucinations/truncations than one giant scene)
  4. Each scene is self-healed up to N times: syntax check -> render -> ACTUAL
     duration check via ffprobe (this is what fixes the "1 second" problem —
     we don't just trust the LLM's promise of duration, we measure it)
  5. Scenes are concatenated with real crossfade transitions (ffmpeg xfade)
  6. Optional cinematic color grade + vignette + letterbox + background music

Requirements (put in requirements.txt / install manually):
    streamlit
    google-genai
    pillow
    pdfplumber
    pymupdf          # fallback: rasterize scanned/image-only PDF pages
Also required on the SYSTEM (not pip):
    manim             (pip install manim, plus its system deps: pango/cairo)
    ffmpeg            (must be on PATH — used for concat, grading, duration probing)

Run:
    streamlit run cinematic_animator_app.py
"""

import io
import json
import os
import py_compile
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image
from google import genai

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Cinematic STEM Animator", page_icon="🎬", layout="wide")
st.title("🎬 AI Cinematic STEM Animator")
st.caption(
    "Text / image / PDF → storyboard → multi-scene Manim render → cinematic cut. "
    "Manim can't be photoreal (it's a vector engine, not a ray-tracer) — this "
    "pipeline pushes it to its cinematic ceiling: depth, glow, camera motion, "
    "grading, music, crossfades."
)

# ----------------------------------------------------------------------------
# Sidebar configuration
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Gemini API Key:", type="password")

MODEL_OPTIONS = [
    "gemini-3.5-flash",        # GA, fastest current flagship-tier flash
    "gemini-3.1-pro-preview",  # strongest reasoning, best for dense/tricky problems
    "gemini-2.5-pro",          # stable fallback (retires 16 Oct 2026)
    "gemini-2.5-flash",        # stable fallback (retires 16 Oct 2026)
]
model_name = st.sidebar.selectbox(
    "Gemini model", MODEL_OPTIONS,
    help="3.5-flash = fast & cheap. 3.1-pro-preview = best reasoning for hard/cramped "
         "questions. The 2.5 models are stable but Google retires them 16 Oct 2026.",
)

quality = st.sidebar.selectbox("Render Quality", ["Low (-ql, fast draft)", "Medium (-qm)", "High (-qh, 1080p60)"])
quality_flag = "-ql" if "Low" in quality else ("-qm" if "Medium" in quality else "-qh")

target_duration = st.sidebar.slider("Target total duration (seconds)", 0, 120, 45, step=5)
style_theme = st.sidebar.selectbox(
    "Visual theme",
    ["Deep Space (navy/cyan glow)", "Blueprint (dark slate/amber)", "Aurora (violet/teal gradient)"],
)
use_crossfade = st.sidebar.checkbox("Cinematic crossfade transitions", value=True)
transition_len = st.sidebar.slider("Crossfade length (s)", 0.2, 1.5, 0.6, step=0.1) if use_crossfade else 0.0
use_cinematic_grade = st.sidebar.checkbox("Cinematic color grade + vignette + letterbox", value=True)
max_heal_attempts = st.sidebar.slider("Max self-heal attempts per scene", 1, 4, 3)
music_file = st.sidebar.file_uploader("Optional background music (mp3)", type=["mp3"])
music_volume = st.sidebar.slider("Music volume", 0.0, 1.0, 0.12) if music_file else 0.0

THEME_PALETTES = {
    "Deep Space (navy/cyan glow)": dict(bg="#03060f", primary="#38f2ff", accent="#ff5da2", text="#f5f9ff"),
    "Blueprint (dark slate/amber)": dict(bg="#0b1420", primary="#ffb454", accent="#5ac8ff", text="#eef3f8"),
    "Aurora (violet/teal gradient)": dict(bg="#0a0518", primary="#7ee8fa", accent="#c17cff", text="#f7f2ff"),
}
palette = THEME_PALETTES[style_theme]

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
input_type = st.radio("Input Type:", ["Text Question", "Image / Screenshot", "PDF"], horizontal=True)

question_text, question_image, pdf_bytes = "", None, None
if input_type == "Text Question":
    question_text = st.text_area(
        "Question Text:", height=100,
        placeholder="e.g., Draw a 3D helical trajectory of a charged particle in a uniform magnetic field.",
    )
elif input_type == "Image / Screenshot":
    uploaded_img = st.file_uploader("Upload Question Image", type=["jpg", "png", "jpeg"])
    if uploaded_img:
        question_image = Image.open(uploaded_img)
        st.image(question_image, caption="Uploaded Image", width=350)
else:
    uploaded_pdf = st.file_uploader("Upload Question PDF", type=["pdf"])
    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.read()
        st.success(f"Loaded PDF ({len(pdf_bytes) / 1024:.0f} KB)")


# ============================================================================
# Ingestion helpers
# ============================================================================
def extract_pdf_pages(raw_bytes: bytes, max_pages: int = 5):
    """Return list of (page_text, PIL.Image or None). Falls back to
    rasterizing a page to an image when it has ~no extractable text
    (i.e. it's a scan/screenshot embedded in the PDF)."""
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                text = (page.extract_text() or "").strip()
                pages.append((text, None))
    except Exception:
        pages = []

    needs_raster = not pages or all(len(t) < 20 for t, _ in pages)
    if needs_raster:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = []
            for i in range(min(len(doc), max_pages)):
                pix = doc[i].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                pages.append(("", img))
        except Exception as e:
            st.warning(f"Could not rasterize PDF pages ({e}). Falling back to text-only extraction.")
    return pages


def build_question_context(client: genai.Client, model: str) -> str:
    """Normalize whatever the user gave us into one rich text description
    Gemini can use to write a storyboard."""
    prompt = (
        "You are reading a STEM question for a student. Extract and restate: "
        "(1) the exact problem statement, (2) all given data/values, (3) what "
        "is being asked, (4) the core concept(s) involved, (5) the key physical "
        "or mathematical objects that must appear in a 3D visualization "
        "(e.g. vectors, trajectories, surfaces, fields, geometric solids). "
        "Be precise and complete — this will drive an animation storyboard."
    )
    if question_text.strip():
        return f"QUESTION (typed by student):\n{question_text.strip()}"

    if question_image is not None:
        resp = client.models.generate_content(model=model, contents=[prompt, question_image])
        return f"QUESTION (extracted from image):\n{resp.text.strip()}"

    if pdf_bytes is not None:
        pages = extract_pdf_pages(pdf_bytes)
        contents = [prompt]
        combined_text = []
        for text, img in pages:
            if text:
                combined_text.append(text)
            if img is not None:
                contents.append(img)
        if combined_text:
            contents.append("Extracted PDF text:\n" + "\n---\n".join(combined_text))
        if len(contents) == 1:
            raise ValueError("Could not extract any content from the PDF.")
        resp = client.models.generate_content(model=model, contents=contents)
        return f"QUESTION (extracted from PDF):\n{resp.text.strip()}"

    raise ValueError("No input provided.")


# ============================================================================
# Storyboard generation
# ============================================================================
def sanitize_json(raw_text: str) -> str:
    cleaned = re.sub(r"```(?:json)?\n?", "", raw_text)
    return cleaned.replace("```", "").strip()


def sanitize_code(raw_text: str) -> str:
    cleaned = re.sub(r"```(?:python)?\n?", "", raw_text)
    return cleaned.replace("```", "").strip()


STORYBOARD_SYSTEM_PROMPT = """
You are a cinematic director for short STEM explainer videos rendered in Manim
Community Edition. Given a question, break it into a SEQUENCE OF SCENES that
together explain and visually solve it, like a mini documentary.

Rules:
- Output ONLY valid JSON. No markdown fences, no commentary.
- Return a JSON object: {{"scenes": [ ... ]}}
- Each scene object has: "title" (string), "description" (string, detailed —
  what appears, what happens, in what order, any camera moves), and
  "duration_seconds" (number, 6-15).
- The scenes' duration_seconds MUST sum to approximately {target_duration}
  seconds (within 10%).
- Typical structure: (1) hook/title + setup of the scenario, (2) establish the
  given objects/variables in 3D, (3) core mechanism / derivation / motion,
  (4) result highlighted, (5) short recap. Adapt to the actual question —
  don't force a template if it doesn't fit.
- Each scene must be self-contained enough that an artist could build it
  without needing the other scenes' code, but MUST visually continue the
  same coordinate system / objects (i.e. describe exact axis ranges, object
  names/colors so continuity holds across scenes).
"""


def generate_storyboard(client: genai.Client, model: str, context: str, target_duration: int) -> list:
    prompt = STORYBOARD_SYSTEM_PROMPT.format(target_duration=target_duration)
    resp = client.models.generate_content(model=model, contents=[prompt, context])
    data = json.loads(sanitize_json(resp.text))
    scenes = data["scenes"]
    if not scenes:
        raise ValueError("Storyboard came back empty.")
    return scenes


# ============================================================================
# Per-scene Manim code generation
# ============================================================================
SCENE_SYSTEM_PROMPT = """
You are an expert Manim Community Edition (v0.18+) animator producing ONE
scene of a multi-scene cinematic explainer video.

HARD OUTPUT REQUIREMENTS:
1. Output ONLY valid, runnable Python. No markdown fences, no commentary.
2. Define exactly one class: `class GeneratedScene(ThreeDScene):`
3. Start the file with `from manim import *` and `import numpy as np`.
4. Do NOT use MathTex() or Tex() (LaTeX may be unavailable on the render
   host). Use Text() everywhere, including for axis labels and equations
   written out as plain text (e.g. Text("F = q(v x B)")).
5. THE SCENE MUST LAST AT LEAST {min_duration} SECONDS OF WALL-CLOCK ANIMATION
   TIME. This is the single most important rule. To hit this:
   - Every self.play(...) MUST have an explicit run_time (never rely on the
     1-second default).
   - Add self.wait(...) between beats so the viewer can read/absorb them.
   - Sum your run_time + wait values mentally before finishing — if the sum
     is under {min_duration}, ADD MORE: extra camera moves, staggered
     reveals (lag_ratio), or an extra explanatory beat. Do not pad with a
     single long self.wait() — pad with actual staged animation.

CINEMATIC VISUAL LANGUAGE (use these to make it look premium, not like a
bare-bones diagram):
- Background: fill with the scene's background color {bg_color} and add a
  sparse starfield/particle field — a VGroup of 40-80 small Dots (radius
  0.01-0.03) at random positions with random low opacities (0.1-0.4),
  faded in with a staggered lag_ratio, sitting behind everything else.
- Palette: primary color {primary_color} for the main object being taught,
  accent color {accent_color} for secondary/highlight elements, text color
  {text_color} for all Text(). Use set_color_by_gradient([...]) on key
  titles for a premium look.
- Glow: fake a glow by stacking 2-3 concentric copies of a shape with
  decreasing opacity and increasing scale behind the main object.
- Depth & motion: use self.set_camera_orientation(phi=.., theta=.., zoom=..)
  at scene start, then move the camera at least once with
  self.move_camera(..., run_time=...) or a short
  self.begin_ambient_camera_rotation(rate=0.15) /
  self.stop_ambient_camera_rotation() pair timed to a self.wait().
- Easing: prefer rate_func=smooth or rate_func=there_and_back for organic
  motion instead of the linear default.
- Use Surface (not just wireframes) with checkerboard_colors set to two
  close shades of the primary color for objects that should read as solid
  3D forms (fields, planets, membranes, etc).
- Text hierarchy: titles large (font_size ~48) with Write(), supporting
  labels smaller (font_size ~28) with FadeIn(shift=UP*0.3).

CONTINUITY: {continuity_note}

Return ONLY the Python code.
"""


def generate_scene_code(client: genai.Client, model: str, scene: dict, palette: dict,
                         continuity_note: str, min_duration: float) -> str:
    prompt = SCENE_SYSTEM_PROMPT.format(
        min_duration=round(min_duration, 1),
        bg_color=palette["bg"], primary_color=palette["primary"],
        accent_color=palette["accent"], text_color=palette["text"],
        continuity_note=continuity_note,
    )
    scene_brief = (
        f"SCENE TITLE: {scene['title']}\n"
        f"TARGET DURATION: {scene['duration_seconds']} seconds\n"
        f"DESCRIPTION: {scene['description']}"
    )
    resp = client.models.generate_content(model=model, contents=[prompt, scene_brief])
    return sanitize_code(resp.text)


def estimate_static_duration(code: str) -> float:
    """Rough heuristic sum of run_time=/self.wait(...) numeric literals,
    used only as a quick pre-render sanity gate (the real check is ffprobe
    on the rendered file)."""
    total = 0.0
    for match in re.finditer(r"run_time\s*=\s*([\d.]+)", code):
        total += float(match.group(1))
    for match in re.finditer(r"self\.wait\(\s*([\d.]*)\s*\)", code):
        total += float(match.group(1)) if match.group(1) else 1.0
    return total


# ============================================================================
# Render + self-heal
# ============================================================================
@dataclass
class SceneResult:
    index: int
    title: str
    success: bool = False
    video_path: Optional[str] = None
    actual_duration: float = 0.0
    code: str = ""
    log: list = field(default_factory=list)


def compile_check(code: str, tmpdir: str) -> Optional[str]:
    """Returns error string, or None if it compiles cleanly."""
    tmp_path = os.path.join(tmpdir, "syntax_check.py")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        py_compile.compile(tmp_path, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return str(e)


def run_manim_render(script_path: str, tmpdir: str, quality_flag: str) -> subprocess.CompletedProcess:
    cmd = ["manim", quality_flag, "--media_dir", tmpdir, "--custom_folders",
           "--fps", "30", script_path, "GeneratedScene"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def find_mp4(tmpdir: str) -> Optional[str]:
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def get_video_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrapper=1:nokey=1", path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def render_scene_with_healing(client: genai.Client, model: str, scene: dict, idx: int,
                               palette: dict, continuity_note: str, quality_flag: str,
                               max_attempts: int, status_box) -> SceneResult:
    result = SceneResult(index=idx, title=scene["title"])
    target = float(scene["duration_seconds"])
    min_acceptable = target * 0.6  # accept if we get at least 60% of target

    code = generate_scene_code(client, model, scene, palette, continuity_note, target)
    error_feedback = None

    for attempt in range(1, max_attempts + 1):
        status_box.write(f"Scene {idx + 1} — attempt {attempt}/{max_attempts}…")

        if error_feedback:
            fix_prompt = (
                f"Your previous Manim code for this scene had a problem:\n"
                f"{error_feedback}\n\n"
                f"PREVIOUS CODE:\n{code}\n\n"
                f"Fix it. Keep the same visual intent. Remember: total animation "
                f"time must be at least {target:.1f} seconds (explicit run_time "
                f"and self.wait values). Return ONLY corrected Python code."
            )
            resp = client.models.generate_content(model=model, contents=[fix_prompt])
            code = sanitize_code(resp.text)

        with tempfile.TemporaryDirectory() as scene_tmp:
            compile_err = compile_check(code, scene_tmp)
            if compile_err:
                error_feedback = f"SyntaxError during compile check:\n{compile_err}"
                result.log.append(f"attempt {attempt}: compile error")
                continue

            static_est = estimate_static_duration(code)
            if static_est < min_acceptable:
                error_feedback = (
                    f"Static analysis found only ~{static_est:.1f}s of run_time/wait "
                    f"values, but {target:.1f}s is required. Add more staged "
                    f"animation beats (not one long wait)."
                )
                result.log.append(f"attempt {attempt}: too short on paper ({static_est:.1f}s)")
                continue

            script_path = os.path.join(scene_tmp, f"scene_{idx}.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                proc = run_manim_render(script_path, scene_tmp, quality_flag)
            except subprocess.TimeoutExpired:
                error_feedback = "Render timed out after 600s — simplify the scene (fewer objects/effects)."
                result.log.append(f"attempt {attempt}: timeout")
                continue

            if proc.returncode != 0:
                error_feedback = f"Manim render error:\n{proc.stderr[-3000:]}"
                result.log.append(f"attempt {attempt}: render error")
                continue

            mp4 = find_mp4(scene_tmp)
            if not mp4:
                error_feedback = "Render reported success but no .mp4 was found."
                result.log.append(f"attempt {attempt}: no output file")
                continue

            actual = get_video_duration(mp4)
            if actual < min_acceptable:
                error_feedback = (
                    f"The rendered video is only {actual:.1f}s long but needed "
                    f"~{target:.1f}s. Add substantially more animation: more "
                    f"self.play() beats, longer run_time values, and self.wait() "
                    f"pauses between them. Do not use a single giant self.wait() — "
                    f"stage multiple distinct visual beats."
                )
                result.log.append(f"attempt {attempt}: rendered but too short ({actual:.1f}s)")
                continue

            # success — copy out of the temp dir before it's cleaned up
            persist_dir = tempfile.mkdtemp()
            final_path = os.path.join(persist_dir, f"scene_{idx}.mp4")
            with open(mp4, "rb") as src, open(final_path, "wb") as dst:
                dst.write(src.read())
            result.success = True
            result.video_path = final_path
            result.actual_duration = actual
            result.code = code
            status_box.write(f"✅ Scene {idx + 1} rendered — {actual:.1f}s")
            return result

    status_box.write(f"⚠️ Scene {idx + 1} did not reach target duration after {max_attempts} attempts — using best effort.")
    result.code = code
    return result


# ============================================================================
# ffmpeg concatenation + cinematic post-processing
# ============================================================================
def concat_with_crossfade(video_paths: list, durations: list, transition: float, tmpdir: str) -> str:
    if len(video_paths) == 1:
        return video_paths[0]

    inputs = []
    for p in video_paths:
        inputs += ["-i", p]

    filter_parts = []
    prev_label = "0:v"
    running_duration = durations[0]
    for i in range(1, len(video_paths)):
        offset = max(running_duration - transition, 0.01)
        out_label = f"v{i}"
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={transition}:offset={offset:.2f}[{out_label}]"
        )
        running_duration = running_duration + durations[i] - transition
        prev_label = out_label

    filter_complex = ";".join(filter_parts)
    out_path = os.path.join(tmpdir, "concat_crossfade.mp4")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
           "-map", f"[{prev_label}]", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path


def concat_simple(video_paths: list, tmpdir: str) -> str:
    list_path = os.path.join(tmpdir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
    out_path = os.path.join(tmpdir, "concat_simple.mp4")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path


def apply_cinematic_grade(in_path: str, tmpdir: str) -> str:
    out_path = os.path.join(tmpdir, "graded.mp4")
    vf = (
        "eq=contrast=1.08:saturation=1.15:brightness=0.01,"
        "vignette=PI/4,"
        "pad=iw:ih*1.14:0:(oh-ih)/2:black"  # letterbox bars
    )
    cmd = ["ffmpeg", "-y", "-i", in_path, "-vf", vf, "-c:v", "libx264",
           "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path


def mux_music(video_path: str, music_bytes: bytes, volume: float, tmpdir: str) -> str:
    music_path = os.path.join(tmpdir, "music.mp3")
    with open(music_path, "wb") as f:
        f.write(music_bytes)
    out_path = os.path.join(tmpdir, "final_with_music.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", f"[1:a]volume={volume}[a]",
        "-map", "0:v", "-map", "[a]", "-shortest",
        "-c:v", "copy", "-c:a", "aac", out_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path


# ============================================================================
# Main pipeline
# ============================================================================
if st.button("🚀 Generate Cinematic Animation", use_container_width=True):
    if not api_key:
        st.error("⚠️ Please enter a valid Gemini API Key in the sidebar.")
    elif not question_text.strip() and question_image is None and pdf_bytes is None:
        st.error("⚠️ Please provide text, an image, or a PDF.")
    else:
        client = genai.Client(api_key=api_key)
        output_root = tempfile.mkdtemp()

        try:
            with st.status("🧠 Reading the question…", expanded=True) as status:
                context = build_question_context(client, model_name)
                st.write(context[:600] + ("…" if len(context) > 600 else ""))
                status.update(label="🎬 Writing storyboard…")

                scenes = generate_storyboard(client, model_name, context, target_duration)
                status.update(label=f"📋 Storyboard: {len(scenes)} scenes", state="running")
                for i, sc in enumerate(scenes):
                    st.write(f"**Scene {i + 1}: {sc['title']}** — ~{sc['duration_seconds']}s")

            scene_results = []
            continuity_note = "This is scene 1 — establish the coordinate system and objects."
            for i, sc in enumerate(scenes):
                box = st.status(f"Rendering scene {i + 1}/{len(scenes)}: {sc['title']}", expanded=True)
                res = render_scene_with_healing(
                    client, model_name, sc, i, palette, continuity_note,
                    quality_flag, max_heal_attempts, box,
                )
                scene_results.append(res)
                continuity_note = (
                    f"This continues directly from scene {i + 1} ('{sc['title']}'). "
                    f"Keep the same axis ranges, object names, and color palette "
                    f"for visual continuity."
                )
                box.update(state="complete" if res.success else "error")

            usable = [r for r in scene_results if r.success and r.video_path]
            if not usable:
                st.error("❌ No scene rendered successfully. Check the logs below.")
                for r in scene_results:
                    with st.expander(f"Scene {r.index + 1} log"):
                        st.write(r.log)
                        if r.code:
                            st.code(r.code, language="python")
            else:
                with st.status("🎞️ Assembling final cut…", expanded=True) as status:
                    paths = [r.video_path for r in usable]
                    durations = [r.actual_duration for r in usable]

                    if use_crossfade and len(paths) > 1:
                        try:
                            combined = concat_with_crossfade(paths, durations, transition_len, output_root)
                        except subprocess.CalledProcessError:
                            st.warning("Crossfade concat failed, falling back to simple concat.")
                            combined = concat_simple(paths, output_root)
                    else:
                        combined = concat_simple(paths, output_root) if len(paths) > 1 else paths[0]

                    if use_cinematic_grade:
                        status.update(label="🎨 Applying cinematic grade…")
                        combined = apply_cinematic_grade(combined, output_root)

                    if music_file is not None:
                        status.update(label="🎵 Mixing background music…")
                        combined = mux_music(combined, music_file.getvalue(), music_volume, output_root)

                    status.update(label="✅ Done", state="complete")

                total_actual = get_video_duration(combined)
                st.success(f"🎉 Rendered {len(usable)}/{len(scenes)} scenes — final duration {total_actual:.1f}s")
                st.video(combined)
                with open(combined, "rb") as f:
                    st.download_button("⬇️ Download MP4", f, file_name="cinematic_animation.mp4", mime="video/mp4")

                if len(usable) < len(scenes):
                    st.warning(
                        f"{len(scenes) - len(usable)} scene(s) never hit the duration target and were "
                        f"skipped — the video is shorter than requested. See logs below."
                    )
                for r in scene_results:
                    with st.expander(f"Scene {r.index + 1} — {r.title} ({'ok' if r.success else 'failed'})"):
                        st.write(r.log)
                        if r.code:
                            st.code(r.code, language="python")

        except Exception as e:
            st.error(f"Application Error: {e}")
