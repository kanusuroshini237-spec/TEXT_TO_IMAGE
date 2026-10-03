"""
Text-to-Image Generator
Generates images from text prompts using Hugging Face's Inference API
(FLUX.1-schnell / Stable Diffusion XL). Needs a FREE Hugging Face token.
"""
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
st.set_page_config(page_title="Text to Image", page_icon="🎨", layout="centered")

# ---------------- Custom CSS ----------------
CUSTOM_CSS = """
<style>

:root {
    --ink: #1d2140;
    --muted: #5d6385;
    --mist: #edeff7;
    --card: #ffffff;
    --line: #d9dcec;
    --indigo: #3f4bd8;
    --indigo-dark: #2f39b0;
    --marigold: #f2b134;
}

/* Base */
html, body, .stApp, [class*="css"] {
    font-family: system-ui, sans-serif;
}
.stApp {
    background: var(--mist);
    color: var(--ink);
}
.block-container {
    max-width: 760px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Headline */
.stApp h1 {
    font-family: system-ui, sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    color: var(--ink);
    padding-bottom: 0.25rem;
    border-bottom: 4px solid var(--marigold);
    display: inline-block;
}
.subtitle {
    color: var(--muted);
    font-size: 1.05rem;
    margin: 0.75rem 0 1.75rem;
    max-width: 52ch;
}

/* Labels */
.stApp label p {
    color: var(--ink);
    font-weight: 600;
    font-size: 0.95rem;
}

/* Inputs */
.stTextArea textarea,
.stTextInput input {
    background: var(--card);
    color: var(--ink);
    border: 1.5px solid var(--line);
    border-radius: 12px;
    font-size: 1rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--indigo);
    box-shadow: 0 0 0 3px rgba(63, 75, 216, 0.18);
}
.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #9097b8;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background: var(--card);
    border: 1.5px solid var(--line);
    border-radius: 10px;
    color: var(--ink);
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    font-family: system-ui, sans-serif;
    font-weight: 500;
    border-radius: 12px;
    padding: 0.65rem 1.6rem;
    transition: transform 0.1s ease, background 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: var(--indigo);
    color: #fff;
    border: none;
    box-shadow: 0 6px 16px rgba(63, 75, 216, 0.28);
}
.stButton > button[kind="primary"]:hover {
    background: var(--indigo-dark);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]:active { transform: translateY(0); }
.stDownloadButton > button {
    background: var(--card);
    color: var(--ink);
    border: 1.5px solid var(--ink);
}
.stDownloadButton > button:hover {
    background: var(--ink);
    color: #fff;
}
button:focus-visible {
    outline: 3px solid var(--marigold) !important;
    outline-offset: 2px;
}

/* Generated image: framed like a print */
[data-testid="stImage"] {
    background: var(--card);
    padding: 12px 12px 6px;
    border-radius: 16px;
    border: 1px solid var(--line);
    box-shadow: 0 14px 34px rgba(29, 33, 64, 0.14);
    margin: 1.25rem 0 1rem;
}
[data-testid="stImage"] img { border-radius: 8px; }
[data-testid="stImage"] [data-testid="stImageCaption"],
[data-testid="stImage"] figcaption {
    color: var(--muted);
    font-size: 0.9rem;
    text-align: left;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--ink);
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stMarkdown p {
    color: #eceefb;
}
[data-testid="stSidebar"] h2 {
    font-family: system-ui, sans-serif;
    font-size: 1.2rem;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #2a2f5a;
    border-color: #3b4180;
    color: #fff;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #2a2f5a;
    border-color: #3b4180;
    color: #fff;
}
[data-testid="stSidebar"] div[data-baseweb="select"] svg { fill: #fff; }

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--line);
}

/* Spinner */
.stSpinner > div > div { border-top-color: var(--indigo) !important; }

/* Mobile */
@media (max-width: 640px) {
    .block-container { padding-top: 2rem; }
    .stApp h1 { font-size: 1.8rem; }
    .stButton > button,
    .stDownloadButton > button { width: 100%; }
}

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODELS = {
    "FLUX.1 Schnell (fast, high quality)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
}

STYLES = {
    "None": "",
    "Photorealistic": ", photorealistic, 8k, highly detailed, sharp focus",
    "Anime": ", anime style, vibrant colors, studio ghibli inspired",
    "Digital Art": ", digital art, trending on artstation, concept art",
    "Oil Painting": ", oil painting, textured brush strokes, classical art",
    "3D Render": ", 3d render, octane render, soft lighting",
}

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
token = st.sidebar.text_input(
    "Hugging Face Token",
    value=os.getenv("HF_TOKEN", ""),
    type="password",
    help="Get a free token at https://huggingface.co/settings/tokens",
)
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
style = st.sidebar.selectbox("Style", list(STYLES.keys()))
size = st.sidebar.selectbox("Size", ["1024x1024", "768x768", "512x512", "1024x768", "768x1024"])
width, height = map(int, size.split("x"))

# ---------------- Main ----------------
st.title("🎨 Text to Image Generator")
st.markdown(
    '<p class="subtitle">Describe an image and let AI draw it.</p>',
    unsafe_allow_html=True,
)

prompt = st.text_area(
    "Your prompt",
    placeholder="A cozy tea stall on a rainy evening in Hyderabad, warm lights, cinematic",
    height=100,
)
negative = st.text_input("Negative prompt (optional)", placeholder="blurry, low quality, distorted")

if st.button("✨ Generate", type="primary"):
    if not token:
        st.error("Please add your Hugging Face token in the sidebar (or in a .env file).")
    elif not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        client = InferenceClient(token=token)
        full_prompt = prompt.strip() + STYLES[style]
        kwargs = {"width": width, "height": height}
        if negative.strip():
            kwargs["negative_prompt"] = negative.strip()

        with st.spinner("Generating image... (10-60 seconds)"):
            try:
                image = client.text_to_image(full_prompt, model=MODELS[model_name], **kwargs)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.info(
                    "Common fixes: check your token, wait a minute if the model is loading, "
                    "or try the other model / a smaller size."
                )
                st.stop()

        st.image(image, caption=prompt)
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button("⬇️ Download PNG", buf.getvalue(), "generated.png", "image/png")