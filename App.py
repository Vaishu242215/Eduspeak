import streamlit as st
import fitz  # PyMuPDF for PDF extraction
from transformers import pipeline
import torch
from googletrans import Translator   # ✔ FREE unlimited translation
from gtts import gTTS
import os
import tempfile

# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------
def chunk_text(text, max_tokens=500):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

# ---------------------------------------------------------
# SUMMARIZATION → BART
# ---------------------------------------------------------
def summarize_text(text):
    try:
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        text_chunks = chunk_text(text, max_tokens=500)
        summaries = [
            summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            for chunk in text_chunks
        ]
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return text

# ---------------------------------------------------------
# TRANSLATION → googletrans (FREE)
# ---------------------------------------------------------
def translate_text(text, target_language="ta"):
    try:
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return "Translation failed."

# ---------------------------------------------------------
# TEXT TO SPEECH
# ---------------------------------------------------------
def text_to_speech(text, language="ta"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_audio.name)
            return temp_audio.name
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="EduSpeak - Summarizer & Translator", layout="wide")
    st.title("EduSpeak: Bridging Education & Regional Languages")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    target_language = st.selectbox("Choose Translation Language", ["ta", "kn", "hi", "te", "ml"], index=0)

    if uploaded_file and st.button("Process PDF"):
        st.subheader("Extracting Text...")
        text = extract_text_from_pdf(uploaded_file)

        if text:
            st.success("Text extracted successfully!")

            st.subheader("Summarizing Content...")
            summary = summarize_text(text)
            st.write(summary)

            st.subheader("Translating to Selected Language...")
            translated_text = translate_text(summary, target_language)
            st.write(translated_text)

            st.subheader("Generating Speech...")
            audio_file = text_to_speech(translated_text, target_language)

            if audio_file:
                st.audio(audio_file, format='audio/mp3')
                st.download_button(
                    "⬇ Download Audio",
                    open(audio_file, "rb"),
                    file_name="translated_audio.mp3",
                    mime="audio/mp3"
                )

if __name__ == "__main__":
    main()

