import os
os.environ["TORCH_DISABLE_RETRY_MODULE_LOOKUP"] = "1"  # Prevents torch class path errors
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Disable file watcher to prevent crash

import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import re
import nltk
from nltk.corpus import words

# Download the English word list (only once)
nltk.download('words')
english_words = set(words.words())

# Function to check prompt validity
def is_meaningful_prompt(prompt):
    prompt = prompt.strip()
    if len(prompt) < 5:
        st.warning("Prompt is too short.")
        return False

    tokens = re.findall(r'\b\w+\b', prompt.lower())
    if not tokens:
        st.warning("Prompt has no recognizable words.")
        return False

    real_word_count = sum(1 for token in tokens if token in english_words)
    ratio = real_word_count / len(tokens)

    if ratio >= 0.5:
        return True
    else:
        st.warning("Prompt seems to be gibberish.")
        return False

# Streamlit Title
st.title("🎨 AI Image Generator (Stable Diffusion)")

# User input
prompt = st.text_input("📝 Enter a prompt to generate an image:")

# Load the model only once
@st.cache_resource
def load_pipeline():
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
    
    # Check if CUDA is available and use GPU if possible, otherwise fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)  # Move to appropriate device (GPU/CPU)
    return pipe

pipe = load_pipeline()

# Button to generate image
if st.button("🚀 Generate Image"):
    if is_meaningful_prompt(prompt):
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="✅ Generated Image", use_column_width=True)
    else:
        st.info("❗ Please enter a more descriptive prompt.")


