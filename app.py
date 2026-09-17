import os
import re
import subprocess
import tempfile
import streamlit as st
from PIL import Image
from google import genai

st.set_page_config(page_title="AI 3D Concept Animator", page_icon="🎬", layout="wide")

st.title("🎬 AI 3D STEM Concept Animator")
st.markdown("Automated 3D rendering pipeline powered by Gemini & Manim Engine.")

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Gemini API Key:", type="password")
model_name = st.sidebar.selectbox(
    "Gemini model",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    help="Flash = faster/cheaper. Pro = slightly more accurate on dense/cramped pages.",
)
quality = st.sidebar.selectbox("Render Quality", ["Low (-ql, Fast)", "Medium (-qm)", "High (-qh, 1080p)"])
quality_flag = "-ql" if "Low" in quality else ("-qm" if "Medium" in quality else "-qh")

input_type = st.radio("Input Type:", ["Text Question", "Image / Screenshot"], horizontal=True)

question_text = ""
question_image = None

if input_type == "Text Question":
    question_text = st.text_area("Question Text:", height=100, 
                                 placeholder="e.g., Draw a 3D helical trajectory of a charged particle moving in a uniform magnetic field.")
else:
    uploaded_file = st.file_uploader("Upload Question Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        question_image = Image.open(uploaded_file)
        st.image(question_image, caption="Uploaded Image", width=350)

SYSTEM_PROMPT = """
You are an expert 3D animator and mathematician specializing in Manim Community Edition (v0.18+).
Your task is to write executable Python code using Manim to construct a 3D animation explaining the concept.

CRITICAL CODE REQUIREMENTS:
1. Output ONLY valid, runnable Python code. Do not include introductory text or markdown wrappers.
2. Define a single scene class named `GeneratedScene(ThreeDScene)`.
3. Use 3D elements like `ThreeDAxes`, `Sphere`, `ParametricFunction`, `Line3D`, `Arrow3D`, `Text`, or `MathTex`.
4. Set up camera positions: `self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)`.
5. Keep animations under 10 seconds.
6. Start output directly with `from manim import *`.
"""

def sanitize_code(raw_text):
    cleaned = re.sub(r'```(?:python)?\n?', '', raw_text)
    return cleaned.replace('```', '').strip()

def run_manim_render(script_path, tmpdirname, quality_flag):
    cmd = ["manim", quality_flag, "--media_dir", tmpdirname, "--custom_folders", script_path, "GeneratedScene"]
    return subprocess.run(cmd, capture_output=True, text=True)

if st.button("🚀 Render 3D Animation", use_container_width=True):
    if not api_key:
        st.error("⚠️ Please enter a valid Gemini API Key in the sidebar.")
    elif not question_text and not question_image:
        st.error("⚠️ Please provide text or upload an image.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            
            with st.spinner("🧠 Analyzing question & constructing 3D scene parameters..."):
                contents = [SYSTEM_PROMPT]
                if question_text:
                    contents.append(f"Question: {question_text}")
                if question_image:
                    contents.append(question_image)
                    contents.append("Construct a 3D animation explaining this image.")

                response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
                manim_code = sanitize_code(response.text)

            # Render Phase with Self-Healing Loop
            with tempfile.TemporaryDirectory() as tmpdirname:
                script_path = os.path.join(tmpdirname, "scene.py")
                
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(manim_code)

                with st.spinner("🎥 Rendering 3D animation (Attempt 1)..."):
                    result = run_manim_render(script_path, tmpdirname, quality_flag)

                # Self-Healing Loop: Auto-fix code if Manim returns an error
                if result.returncode != 0:
                    st.warning("⚠️ Initial code encountered a rendering error. Triggering AI Self-Healing Loop...")
                    fix_prompt = f"""
                    The following Manim Python code produced an error during execution:
                    
                    CODE:
                    {manim_code}
                    
                    ERROR LOG:
                    {result.stderr}
                    
                    Please fix the error and return ONLY valid, updated Python code for `GeneratedScene(ThreeDScene)`.
                    """
                    
                    fix_response = client.models.generate_content(model='gemini-2.5-flash', contents=[SYSTEM_PROMPT, fix_prompt])
                    manim_code = sanitize_code(fix_response.text)
                    
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(manim_code)

                    with st.spinner("🎥 Re-rendering fixed 3D animation (Attempt 2)..."):
                        result = run_manim_render(script_path, tmpdirname, quality_flag)

                st.expander("📄 View Executed Script").code(manim_code, language="python")

                if result.returncode != 0:
                    st.error("❌ Render failed after self-healing attempt:")
                    st.code(result.stderr, language="text")
                else:
                    video_file = None
                    for root, dirs, files in os.walk(tmpdirname):
                        for file in files:
                            if file.endswith(".mp4"):
                                video_file = os.path.join(root, file)
                                break
                    
                    if video_file and os.path.exists(video_file):
                        st.success("🎉 3D Animation Generated Successfully!")
                        st.video(video_file)
                    else:
                        st.error("Render finished, but .mp4 output file could not be located.")

        except Exception as e:
            st.error(f"Application Error: {str(e)}")
