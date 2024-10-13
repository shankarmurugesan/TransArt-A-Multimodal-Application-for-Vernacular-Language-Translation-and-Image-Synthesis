import os
import io
import requests
import gradio as gr
from groq import Groq
from transformers import MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw
import joblib
import time
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import openai
import torch
import warnings
from huggingface_hub import InferenceApi

# Detect if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up Groq API key
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Set your Hugging Face API key
os.environ['HF_API_KEY']
api_key = os.getenv('HF_API_KEY')
if api_key is None:
    raise ValueError("Hugging Face API key is not set. Please set it in your environment.")

# Set OpenAI API key for text generation
openai.api_key = os.getenv('OPENAI_API_KEY')

headers = {"Authorization": f"Bearer {api_key}"}

# Load GPT-Neo for creative text generation
text_generation_model_name = "EleutherAI/gpt-neo-1.3B"
text_generation_model = AutoModelForCausalLM.from_pretrained(text_generation_model_name).to(device)
text_generation_tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name)

# Add padding token to GPT-Neo tokenizer if not present
if text_generation_tokenizer.pad_token is None:
    text_generation_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define the API URL for image generation
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"

# Load the trained sentiment analysis model and preprocessing steps
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load('model.pkl')

# Function to query Hugging Face API
def query(payload, max_retries=5):
    for attempt in range(max_retries):
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 503:
            print(f"Model is still loading, retrying... Attempt {attempt + 1}/{max_retries}")
            estimated_time = min(response.json().get("estimated_time", 60), 60)
            time.sleep(estimated_time)
            continue

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None

        return response.content

    print(f"Failed to generate image after {max_retries} attempts.")
    return None

# Function to generate image
def generate_image(prompt):
    image_bytes = query({"inputs": prompt})

    if image_bytes is None:
        error_img = Image.new('RGB', (300, 300), color=(255, 0, 0))
        d = ImageDraw.Draw(error_img)
        d.text((10, 150), "Image Generation Failed", fill=(255, 255, 255))
        return error_img

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        print(f"Error: {e}")
        error_img = Image.new('RGB', (300, 300), color=(255, 0, 0))
        d = ImageDraw.Draw(error_img)
        d.text((10, 150), "Invalid Image Data", fill=(255, 255, 255))
        return error_img

# Tamil Audio to Tamil text
def transcribe_audio(audio_path):
    if audio_path is None:
        return "Please upload an audio file."
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        return transcription.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Transliterate Romanized Tamil (in English letters) to Tamil script
def transliterate_to_tamil(romanized_text):
    try:
        # Step 1: Normalize the input for better transliteration results
        romanized_text = romanized_text.strip().lower()  # Remove extra spaces and convert to lowercase

        # Step 2: Handle common punctuation that might interrupt transliteration
        romanized_text = romanized_text.replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ")

        # Step 3: Apply ITRANS transliteration
        tamil_text = transliterate(romanized_text, sanscript.ITRANS, sanscript.TAMIL)

        return tamil_text
    except Exception as e:
        return f"An error occurred during transliteration: {str(e)}"

# Function to translate Tamil text to English using deep-translator
def translate_tamil_to_english(tamil_text):
    if not tamil_text:
        return "Please provide text to translate."
    try:
        translator = GoogleTranslator(source='ta', target='en')
        translated_text = translator.translate(tamil_text)

        # Predict sentiment from translated text
        sentiment_result = predict_sentiment(translated_text)

        return translated_text, sentiment_result, translated_text
    except Exception as e:
        return f"An error occurred during translation: {str(e)}", None, None

# Function to predict sentiment from English text
def predict_sentiment(english_text):
    if not english_text:
        return "No text provided for sentiment analysis."
    try:
        sentiment = model.predict([english_text])[0]
        return f"Sentiment: {sentiment}"
    except Exception as e:
        return f"An error occurred during sentiment prediction: {str(e)}"

# Generate creative text based on the translated English text
def generate_creative_text(english_text):
    if not english_text:
        return "Please provide text to generate creative content."

    try:
        inputs = text_generation_tokenizer(english_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Set parameters to control the output and avoid repetition
        generated_tokens = text_generation_model.generate(
            **inputs,
            max_length=60,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            early_stopping=True
        )

        creative_text = text_generation_tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        return creative_text

    except Exception as e:
        return f"An error occurred during text generation: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='color: #4CAF50;'>🎙️ Tamil Audio Transcription, Translation, Sentiment Prediction, Creative Text Generation, and Image Generation</h1>
        <p style='color: #000080;'>Upload an audio file to get the Tamil transcription, edit the transcription or type Romanized Tamil to convert it to Tamil script, translate it to English, predict the sentiment of the translated text, generate creative English text, and generate an image.</p>
        """
    )

    # Input for audio file
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio File")
        transcribe_button = gr.Button("Transcribe Audio", elem_id="transcribe_btn")

    # Output field for Tamil transcription with ability to edit or type Romanized Tamil
    transcription_output = gr.Textbox(label="Transcription (Tamil or Romanized Tamil)", interactive=True ,elem_id="transcription_output")

    # Button for transliterating Romanized Tamil to Tamil script
    transliterate_button = gr.Button("Convert to Tamil Script", elem_id="transliterate_btn")

    # Input field for Tamil text and translate button
    with gr.Row():
        translate_button = gr.Button("Translate to English", elem_id="translate_btn")

    # Output field for English translation
    translation_output = gr.Textbox(label="Translation (English)", elem_id="translation_output")

    # Output field for sentiment prediction
    sentiment_output = gr.Textbox(label="Sentiment", elem_id="sentiment_output")

    # Button to generate creative text
    creative_text_button = gr.Button("Generate Creative Text", elem_id="creative_btn")

    # Output field for creative text
    creative_text_output = gr.Textbox(label="Creative Text", elem_id="creative_output")

    # Button to generate image
    generate_button = gr.Button("Generate Image", elem_id="generate_btn")

    # Output field for image file
    image_output = gr.Image(label="Generated Image")

    # Define variable to hold the translated English text
    translated_text_var = gr.State()

    # Define button click actions
    transcribe_button.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=transcription_output,
    )

    transliterate_button.click(
        fn=transliterate_to_tamil,
        inputs=transcription_output,
        outputs=transcription_output,
    )

    translate_button.click(
        fn=translate_tamil_to_english,
        inputs=transcription_output,
        outputs=[translation_output, sentiment_output, translated_text_var],
    )

    creative_text_button.click(
        fn=generate_creative_text,
        inputs=translated_text_var,
        outputs=creative_text_output,
    )

    generate_button.click(
        fn=generate_image,
        inputs=translated_text_var,
        outputs=image_output,
    )

# Apply custom CSS
demo.css = """
#transcribe_btn, #transliterate_btn, #translate_btn, #creative_btn, #generate_btn {
    background-color: #05907B; /* Change button color */
    color: white; /* Change text color */
}

#translation_output,#transcription_output, #sentiment_output, #creative_output {
    background-color: #f0f8ff; /* Change background color of text areas */
}

h1 {
    color: #4CAF50; /* Main heading color */
}

p {
    color: #000080; /* Plain text color */
}

/* Add thick border to entire app */
.gradio-container {
    border: 5px solid #05907B; /* Thick border color */
    padding: 10px; /* Padding inside the border */
    border-radius: 10px; /* Optional: add rounded corners */
}
"""

# Launch the interface and ensure code stops afterward
demo.launch(share=True)
