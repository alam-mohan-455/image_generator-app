import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler 
import re
import nltk
from nltk.corpus import words
from PIL import Image
import os

nltk.download('words')
english_words = set(words.words())

@st.cache_resource
def load_pipeline():
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    return pipe.to("cuda")

def is_meaningful_prompt(prompt):
    prompt = prompt.strip()
    if len(prompt) < 5:
        return False
    tokens = re.findall(r'\b\w+\b', prompt.lower())
    if not tokens:
        return False
    real_word_count = sum(1 for token in tokens if token in english_words)
    ratio = real_word_count / len(tokens)
    return ratio >= 0.5

st.title("🖼️ AI Image Generator with Stable Diffusion")

pipe = load_pipeline()

prompt = st.text_input("Enter your prompt to generate an image:")

if prompt:
    if is_meaningful_prompt(prompt):
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            image_path = "generated_image.png"
            image.save(image_path)
            st.image(image, caption="Generated Image", use_column_width=True)
            with open(image_path, "rb") as file:
                st.download_button("Download Image", file, file_name="generated_image.png")
    else:
        st.warning("The prompt doesn't seem meaningful. Please try again.")

