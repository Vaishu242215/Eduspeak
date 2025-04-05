import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import fitz  # PyMuPDF for PDF extraction 
from transformers import pipeline
import torch
import google.generativeai as genai
from gtts import gTTS
import os
import tempfile

# Configure Gemini API Key
genai.configure(api_key="AIzaSyAE8y8q3PERuJxBqKQO2WFojL0m8P2jiHA")


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Function to split text into smaller chunks
def chunk_text(text, max_tokens=500):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

# Function to summarize text
def summarize_text(text):
    try:
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        text_chunks = chunk_text(text, max_tokens=500)
        summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                     for chunk in text_chunks]
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return text

# Function to translate text
def translate_text(text, target_language="ta"):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = f"Translate the following English text to {target_language}: {text}"
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "Translation failed."
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return "Translation failed."

# Function to convert text to speech
    def text_to_speech(text, language="ta"):
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            temp_audio_path = os.path.join(tempfile.gettempdir(), "output.mp3")
            tts.save(temp_audio_path)
            return temp_audio_path
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            return None


# Streamlit UI
def main():
    st.set_page_config(page_title="EduSpeak - PDF Summarizer & Translator", layout="wide")
    st.title("📚 EduSpeak: Bridging Education & Regional Languages")
    
    uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    target_language = st.selectbox("🌍 Choose Translation Language", ["ta", "kn", "hi", "te", "ml"], index=0)
    
    if uploaded_file and st.button("🔍 Process PDF"):
        st.subheader("🔹 Extracting Text...")
        text = extract_text_from_pdf(uploaded_file)
        
        if text:
            st.success("Text extracted successfully!")
            
            st.subheader("📖 Summarizing Content...")
            summary = summarize_text(text)
            st.write(summary)
            
            st.subheader("🌎 Translating to Selected Language...")
            translated_text = translate_text(summary, target_language)
            st.write(translated_text)
            
            st.subheader("🔊 Generating Speech...")
            audio_file = text_to_speech(translated_text, target_language)
            if audio_file:
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button("⬇ Download Audio", data=audio_bytes, file_name="translated_audio.mp3", mime="audio/mp3")


if __name__ == "__main__":
    main()
