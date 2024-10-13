import os
import io
import requests
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_translator import GoogleTranslator
from PIL import Image

# Set Hugging Face API Key
HF_API_KEY = os.getenv('HF_API_KEY', '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')
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Hugging Face API URL for image generation
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"

# Streamlit App Setup
st.title("Tamil to English Translation, Creative Text, and Image Generation")

# Initialize session state for storing intermediate results
if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = ""
if "creative_text" not in st.session_state:
    st.session_state["creative_text"] = ""

# Section 1: Tamil to English Translation
st.subheader("1. Translate Tamil Text to English")

# Input text area for Tamil text
tamil_text_input = st.text_area("Enter Tamil Text for Translation", key="tamil_text_input")

# Button to trigger translation
if st.button("Translate to English"):
    if tamil_text_input:
        try:
            # Translate Tamil text to English and store in session state
            translator = GoogleTranslator(source='ta', target='en')
            st.session_state["translated_text"] = translator.translate(tamil_text_input)
        except Exception as e:
            st.session_state["translated_text"] = f"An error occurred during translation: {str(e)}"
    else:
        st.session_state["translated_text"] = "Please enter Tamil text for translation."

# Static text field to show the translated English text
st.text_area("Translated English Text", value=st.session_state["translated_text"], height=100, key="translated_output")

# Lazy loading: Define GPT-Neo text generation model and tokenizer, but don't load them until needed
@st.cache_resource
def load_gpt_neo():
    model_name = "distilgpt2"  # Use a smaller, faster model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model, tokenizer

# Section 2: Creative Text Generation
st.subheader("2. Generate Creative Text from English Translation")

# Use the translated English text as input for creative text generation
creative_input = st.session_state["translated_text"]

# Button to trigger creative text generation
if st.button("Generate Creative Text"):
    if creative_input:
        with st.spinner("Generating creative text..."):
            try:
                # Lazy load GPT-Neo model and tokenizer only when needed
                model, tokenizer = load_gpt_neo()
                inputs = tokenizer(creative_input, return_tensors="pt", padding=True, truncation=True)
                generated_tokens = model.generate(
                    **inputs,
                    max_length=60,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    early_stopping=True
                )
                st.session_state["creative_text"] = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
            except Exception as e:
                st.session_state["creative_text"] = f"An error occurred during text generation: {str(e)}"
    else:
        st.session_state["creative_text"] = "No translated text found. Please translate text first."

# Static text field to show the generated creative text
st.text_area("Generated Creative Text", value=st.session_state["creative_text"], height=100, key="creative_output")

# Function to generate image using Hugging Face API
def generate_image(prompt):
    payload = {"inputs": prompt}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
            return image
        else:
            st.write(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.write(f"An error occurred during image generation: {str(e)}")
        return None

# Section 3: Image Generation
st.subheader("3. Generate Image from English Translation")

# Use the translated English text as input for image generation
image_input = st.session_state["translated_text"]

# Button to trigger image generation
if st.button("Generate Image"):
    if image_input:
        image = generate_image(image_input)
        if image:
            st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.write("Image generation failed.")
    else:
        st.write("No translated text found. Please translate text first.")
