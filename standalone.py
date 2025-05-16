import streamlit as st
import streamlit.components.v1 as components
import easyocr
from PIL import Image, UnidentifiedImageError
import fitz  # PyMuPDF
import io
from typing import Optional, List, Tuple, Any
import json # For NLTK nouns if we re-integrate it
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import time
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration # Keep commented if not immediately used by chatbot parts
# import nltk # Keep commented if NLTK features are not immediately re-integrated

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Eduthon Standalone App", layout="wide", initial_sidebar_state="expanded")

# --- Inject Custom CSS for Lexend font for the entire Streamlit app ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Apply Lexend to the main Streamlit app elements */
    html, body, [class*="st-"], .stApp, .stButton>button, .stTextArea textarea, .stTextInput input, .stFileUploader label, .stSelectbox div[data-baseweb="select"] > div, .stAlert, .stMarkdown, .stExpander header, h1, h2, h3, h4, h5, h6, p, div, span, li, label, button, input, select, textarea {
        font-family: 'Lexend', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Global Initializations & Helper Functions ---

# Initialize EasyOCR reader (from readaloud1.py)
@st.cache_resource
def load_ocr_reader() -> easyocr.Reader:
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize EasyOCR reader: {e}")
        st.stop()
ocr_reader_instance = load_ocr_reader()

# Configure Google GenAI (from chatbot.py)
# TODO: Use st.secrets for API keys in a real application
GENAI_API_KEY = 'AIzaSyDLRh5LHcyYpxQx6oHSKlsX_tj1Xap0Ods' 
try:
    genai.configure(api_key=GENAI_API_KEY)
    genai_model_chat = genai.GenerativeModel('gemini-2.0-flash') # For chatbot
    genai_model_therapy = genai.GenerativeModel('gemini-2.0-flash') # For therapy (can be the same or different)
except Exception as e:
    st.error(f"Failed to configure Google GenAI: {e}. Chatbot features may not work.")
    genai_model_chat = None
    genai_model_therapy = None

# Speech Recognizer (from chatbot.py)
@st.cache_resource
def get_speech_recognizer():
    return sr.Recognizer()
speech_r = get_speech_recognizer()

# --- Helper Functions from readaloud1.py ---
def perform_ocr(pil_image: Image.Image) -> str:
    extracted_text = ""
    try:
        img_byte_arr = io.BytesIO()
        image_format = pil_image.format if pil_image.format else 'PNG'
        pil_image.save(img_byte_arr, format=image_format)
        img_bytes = img_byte_arr.getvalue()
        result: List[Tuple[Any, str, Any]] = ocr_reader_instance.readtext(img_bytes)
        if result:
            extracted_text = ' '.join([text[1] for text in result])
    except Exception as e:
        print(f"OCR Error: {e}")
        st.error(f"An error occurred during OCR processing.")
    return extracted_text.strip()

def extract_text_from_pdf(pdf_file_bytes: bytes) -> str:
    full_extracted_text: List[str] = []
    MIN_CHARS_FOR_DIRECT_TEXT = 50
    OCR_DPI = 150
    try:
        pdf_document = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        num_pages = len(pdf_document)
        progress_bar = st.progress(0, text="Starting PDF processing...")
        status_text = st.empty()
        for page_num in range(num_pages):
            current_progress = (page_num + 1) / num_pages
            status_text.text(f"Processing PDF Page {page_num + 1}/{num_pages}...")
            progress_bar.progress(current_progress, text=f"Page {page_num + 1}/{num_pages}")
            page_text_content = ""
            try:
                page = pdf_document.load_page(page_num)
                direct_text = page.get_text("text").strip()
                ocr_text = ""
                if len(direct_text) < MIN_CHARS_FOR_DIRECT_TEXT:
                    status_text.text(f"Page {page_num + 1}/{num_pages} (low text, trying OCR)...")
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    img_bytes = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    ocr_text = perform_ocr(pil_image).strip()
                if ocr_text and len(direct_text) < MIN_CHARS_FOR_DIRECT_TEXT and len(ocr_text) > len(direct_text):
                    page_text_content = ocr_text
                elif direct_text:
                    page_text_content = direct_text
                elif ocr_text:
                    page_text_content = ocr_text
                if page_text_content:
                    full_extracted_text.append(page_text_content)
            except Exception as e_page:
                st.warning(f"Could not process page {page_num + 1} of PDF: {e_page}")
        status_text.text("PDF processing complete.")
        progress_bar.empty()
        pdf_document.close()
    except fitz.errors.FitzError as fe:
        st.error(f"PyMuPDF Error: {fe}. The PDF might be corrupted or password-protected.")
        if 'progress_bar' in locals(): progress_bar.empty()
    except Exception as e_doc:
        st.error(f"General error processing PDF document: {e_doc}")
        if 'progress_bar' in locals(): progress_bar.empty()
    return "\n\n<page_break>\n\n".join(full_extracted_text).strip()

def js_string_escape_ra(s: str) -> str:
    if not s: return ""
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\t", "\\t")
    )

def get_readaloud_html_content(text_content: str, access_key: str) -> str:
    try:
        with open('readaloud1.html', 'r', encoding='utf-8') as file:
            html_template = file.read()
        escaped_text_content = js_string_escape_ra(text_content)
        # Nouns JSON placeholder is removed for simplicity now, can be re-added if NLTK is integrated
        # html_code = html_template.replace('{{ nouns_for_images_json }}', '[]') # Default to empty array
        html_code = html_template.replace('{{ text_content }}', escaped_text_content)
        html_code = html_code.replace('{{ access_key }}', access_key)
        return html_code
    except FileNotFoundError:
        st.error("Error: 'readaloud1.html' not found.")
        return "<p><b>Error: TTS Player template (readaloud1.html) missing.</b></p>"
    except Exception as e:
        st.error(f"Error reading or processing 'readaloud1.html': {e}")
        return f"<p><b>Error loading TTS Player: {e}</b></p>"

def display_readaloud_tts_player(text_content: str, access_key: str):
    # Unsplash API Key from readaloud1.py context
    # TODO: Use st.secrets for API keys
    unsplash_key = 'F4nLejAZww7_NC1DB8SF7pf0CKQLQhr9kBaZ0w9TISI' 
    html_content = get_readaloud_html_content(text_content, unsplash_key)
    components.html(html_content, height=600, scrolling=True)

# --- Helper Functions from chatbot.py (or adapted) ---
def recognize_speech_from_mic_ch() -> str:
    with sr.Microphone() as source:
        speech_r.adjust_for_ambient_noise(source, duration=0.2)
        try:
            st.info("Listening...")
            audio = speech_r.listen(source, timeout=5, phrase_time_limit=10)
            my_text = speech_r.recognize_google(audio)
            my_text = my_text.lower()
            st.success(f"Recognized: {my_text}")
            return my_text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
        except sr.UnknownValueError:
            st.warning("Google Speech Recognition could not understand audio. Please try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred during speech recognition: {e}")
        return ""

def render_chatbot_html_with_text(text_variable):
    try:
        with open("chatbot.html", "r", encoding="utf-8") as file:
            html_code = file.read()
        html_code = html_code.replace("{text_variable}", text_variable) # Ensure placeholder matches chatbot.html
        components.html(html_code, height=400, scrolling=True)
    except FileNotFoundError:
        st.error("Error: 'chatbot.html' not found. Cannot display chatbot response visually.")
    except Exception as e:
        st.error(f"Error loading or processing 'chatbot.html': {e}")

# --- Tab Functions ---

def read_aloud_tab():
    st.header("📖 Reading Assistance Pro (Text-to-Speech)")
    st.markdown("Upload an image or PDF, or type text directly to have it read aloud.")

    if "ra_manual_text_input" not in st.session_state: st.session_state.ra_manual_text_input = ""
    if "ra_image_extracted_text" not in st.session_state: st.session_state.ra_image_extracted_text = ""
    if "ra_pdf_extracted_text" not in st.session_state: st.session_state.ra_pdf_extracted_text = ""
    if "ra_last_uploaded_image_file_id" not in st.session_state: st.session_state.ra_last_uploaded_image_file_id = None
    if "ra_last_uploaded_pdf_file_id" not in st.session_state: st.session_state.ra_last_uploaded_pdf_file_id = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Methods")
        st.session_state.ra_manual_text_input = st.text_area(
            "**1. Type or Paste Text**", value=st.session_state.ra_manual_text_input, height=100, key="ra_manual_text_area"
        )

        st.markdown("**2. Upload an Image (OCR)**")
        uploaded_image_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"], key="ra_image_uploader"
        )
        if uploaded_image_file is not None and uploaded_image_file.file_id != st.session_state.get("ra_last_uploaded_image_file_id"):
            with st.spinner("Extracting text from image..."):
                try:
                    pil_image = Image.open(uploaded_image_file)
                    st.session_state.ra_image_extracted_text = perform_ocr(pil_image)
                    st.session_state.ra_pdf_extracted_text = "" # Clear other sources
                    st.session_state.ra_last_uploaded_image_file_id = uploaded_image_file.file_id
                    st.session_state.ra_last_uploaded_pdf_file_id = None
                    st.success("Image text extracted!")
                except UnidentifiedImageError: st.error("Not a valid image file.")
                except Exception as e: st.error(f"Error processing image: {e}")

        st.markdown("**3. Upload a PDF**")
        uploaded_pdf_file = st.file_uploader(
            "Choose a PDF...", type=["pdf"], key="ra_pdf_uploader"
        )
        if uploaded_pdf_file is not None and uploaded_pdf_file.file_id != st.session_state.get("ra_last_uploaded_pdf_file_id"):
            st.session_state.ra_pdf_extracted_text = extract_text_from_pdf(uploaded_pdf_file.getvalue())
            st.session_state.ra_image_extracted_text = "" # Clear other sources
            st.session_state.ra_last_uploaded_pdf_file_id = uploaded_pdf_file.file_id
            st.session_state.ra_last_uploaded_image_id = None
            st.success("PDF text extracted!")

        if st.button("Clear All Inputs", key="ra_clear_all"):
            st.toast("Clear All Inputs button was clicked! Attempting to clear...", icon="🧹") # Debug toast
            st.session_state.ra_manual_text_input = ""
            st.session_state.ra_image_extracted_text = ""
            st.session_state.ra_pdf_extracted_text = ""
            st.session_state.ra_last_uploaded_image_file_id = None
            st.session_state.ra_last_uploaded_pdf_file_id = None
            st.rerun()
    
    with col2:
        st.subheader("Extracted Text & Player")
        texts_to_combine = []
        if st.session_state.ra_manual_text_input: texts_to_combine.append(f"[Manual Input]\n{st.session_state.ra_manual_text_input}")
        if st.session_state.ra_image_extracted_text: texts_to_combine.append(f"[Image Text]\n{st.session_state.ra_image_extracted_text}")
        if st.session_state.ra_pdf_extracted_text: texts_to_combine.append(f"[PDF Text]\n{st.session_state.ra_pdf_extracted_text}")
        
        final_combined_text_for_tts = "\n\n".join(texts_to_combine).strip()

        if final_combined_text_for_tts:
            st.text_area("Final Text for Player:", value=final_combined_text_for_tts, height=200, disabled=True, key="ra_final_text_ro")
            st.download_button(
                label="📥 Download Combined Text (.txt)", data=final_combined_text_for_tts.encode('utf-8'),
                file_name="readaloud_extracted_text.txt", mime="text/plain", key="ra_download_btn"
            )
            display_readaloud_tts_player(final_combined_text_for_tts, "unused_key_placeholder") # access_key handled in display_readaloud_tts_player
        else:
            st.info("No text provided for the player. Use the input methods on the left.")

def general_chatbot_tab():
    st.header("💬 General Chatbot for Dyslexic Individuals")
    st.markdown("Ask questions and get simplified explanations.")

    if genai_model_chat is None:
        st.error("General Chatbot is unavailable due to GenAI model initialization failure.")
        return

    # Initialize session state for this tab's text input if not already present
    if 'gc_query_text' not in st.session_state:
        st.session_state.gc_query_text = ""

    user_query_from_mic = ""

    col1, col2, col3 = st.columns([2,1,1]) # Adjust column layout for text input and buttons
    with col1:
        # Text input now reads its value from st.session_state.gc_query_text
        text_input_val = st.text_input("Type your query here:", value=st.session_state.gc_query_text, key="gc_text_input_widget")
    with col2:
        if st.button("Submit Text", key="gc_submit_text"):
            st.session_state.gc_query_text = text_input_val # Update state from text input
            # The processing logic below will use st.session_state.gc_query_text
    with col3:
        if st.button("🎤 Start Microphone", key="gc_start_mic"):
            recognized_text = recognize_speech_from_mic_ch()
            if recognized_text:
                st.session_state.gc_query_text = recognized_text # Update state from mic
                # No need to set user_query here, the text_input will update on rerun if needed or we use st.session_state.gc_query_text directly
                st.rerun() # Rerun to update the text_input with recognized speech

    # Process the query from session state
    current_query_to_process = st.session_state.gc_query_text

    if current_query_to_process:
        # Only display processing info and call API if there was an explicit action (button press)
        # This check helps avoid re-processing on every rerun after mic input updates the text field.
        # We rely on the fact that a button press (Submit or Mic) has populated st.session_state.gc_query_text

        # Check if a button was actually pressed to trigger this processing round
        # This logic is a bit tricky with Streamlit's reruns. A more robust way might involve explicit action flags.
        # For now, we assume if current_query_to_process is non-empty AND a relevant button was just pressed (implied by state change), we proceed.
        
        # Let's simplify: process if there's text and a button implies recent interaction
        # A better way might be to check st.form_submit_button if using forms, or more complex state management.

        # If query exists (from text submit or mic), process it.
        # The rerun after mic input will cause this block to execute with the new query.
        st.info(f"Processing query: {current_query_to_process}")
        with st.spinner("Generating response..."):
            prompt = (
                f"{current_query_to_process}"
                f'''I'd like you to act as an educational assistant for a child with dyslexia. Please keep in mind that they might struggle with reading and writing, so it's important to present information in a simple and clear way.'''
                f'''
            Keep the following things in mind:    
        Simple Language: Use short sentences and avoid complex vocabulary. Break down concepts into smaller, easier-to-understand parts.
        Visual Aids: Whenever possible, use pictures, diagrams, or other visual aids to support the explanation. (But for this text-based response, do not attempt to generate image markdown or links).
        Chunking: Present information in small chunks and allow time for the child to process it before moving on.
        Repetition: It's okay to repeat information if needed.
        Positive Reinforcement: Offer encouragement and praise the child's efforts.
        Your response should be limited to around 120-150 words.
        Focus on being clear, concise, and supportive.
        '''
            )
            try:
                response = genai_model_chat.generate_content(prompt)
                render_chatbot_html_with_text(response.text)
                # Optionally clear the query after processing to prevent re-submission on simple rerun
                # st.session_state.gc_query_text = "" 
            except Exception as e:
                st.error(f"Failed to generate response from GenAI: {e}")
    # Removed the warning for empty query as the flow is now different.
    # else: 
    #      st.warning("Please provide a query through text input or microphone.")


@st.cache_resource # Cache FAISS index and vectorizer
def load_therapy_faiss_index():
    try:
        df = pd.read_csv('dataset.csv', encoding='latin1')
        df['Query'] = df['Query'].apply(lambda x: str(x).lower())
        df['Response'] = df['Response'].apply(lambda x: str(x).lower())
        vectorizer = TfidfVectorizer()
        query_vectors = vectorizer.fit_transform(df['Query'])
        dimension = query_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        query_vectors_faiss = query_vectors.toarray().astype('float32')
        index.add(query_vectors_faiss)
        # faiss.write_index(index, 'therapy_index.faiss') # Not needed for each run if static
        return index, vectorizer, df['Response']
    except FileNotFoundError:
        st.error("Therapy Chatbot Error: 'dataset.csv' not found. Retrieval functionality will be limited.")
        return None, None, None
    except Exception as e:
        st.error(f"Error initializing Therapy Chatbot FAISS index: {e}")
        return None, None, None

therapy_faiss_index, therapy_vectorizer, therapy_df_responses = load_therapy_faiss_index()

def therapy_chatbot_tab():
    st.header("❤️‍🩹 Therapy Support Chatbot")
    st.markdown("A supportive space to discuss feelings and find encouragement.")

    if genai_model_therapy is None:
        st.error("Therapy Chatbot is unavailable due to GenAI model initialization failure.")
        return
    if therapy_faiss_index is None:
        st.warning("Therapy Chatbot retrieval features are limited due to an issue loading 'dataset.csv'. It will rely solely on general AI responses.")

    # Initialize session state for this tab's text input
    if 'th_query_text' not in st.session_state:
        st.session_state.th_query_text = ""

    user_query_from_mic = ""

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        text_input_val = st.text_input("Share your thoughts or questions here:", value=st.session_state.th_query_text, key="th_text_input_widget")
    with col2:
        if st.button("Submit Thoughts", key="th_submit_text"):
            st.session_state.th_query_text = text_input_val
    with col3:
        if st.button("🎤 Use Microphone", key="th_start_mic"):
            recognized_text = recognize_speech_from_mic_ch()
            if recognized_text:
                st.session_state.th_query_text = recognized_text
                st.rerun() # Rerun to update text_input

    current_query_to_process = st.session_state.th_query_text

    if current_query_to_process:
        st.info(f"Processing: {current_query_to_process}")
        retrieved_context = ""
        if therapy_faiss_index and therapy_vectorizer and therapy_df_responses is not None:
            try:
                query_vec = therapy_vectorizer.transform([str(current_query_to_process).lower()]).toarray().astype('float32')
                D, I = therapy_faiss_index.search(query_vec, k=1)
                if I.size > 0 and I[0][0] < len(therapy_df_responses):
                    retrieved_context = therapy_df_responses[I[0][0]]
                    st.caption(f"Retrieved context hint: {retrieved_context[:100]}...")
                else:
                    st.caption("No relevant context found in dataset.")
            except Exception as e:
                st.warning(f"Could not retrieve context from dataset: {e}")
        
        with st.spinner("Thinking..."):
            prompt = (
                f"User (who may be feeling vulnerable, seeking support related to dyslexia): {current_query_to_process}\n"
                f"Background context from similar past interactions (if any): {retrieved_context}\n"
                f"Therapist (empathetic, supportive, focused on helping a dyslexic person):\n"
                f'''Focus on building a safe and supportive space.
                Acknowledge the child's feelings and validate their struggles.
                Emphasize that it's a different way of learning, not a disability.
                Showcase successful people with dyslexia.
                Motivate the child by demonstrating achievement is possible.
                Highlight the importance of support and tools.
                Briefly mention resources like audiobooks, specialized tutors, or assistive technologies.
                End on a positive and empowering note.
                Remind the child of their strengths and potential.
                Use positive and affirming language throughout.
                Maintain a conversational and approachable tone.
                Encourage the child to ask questions and express their feelings.
                Give the response in 120 to 150 words and stick to the query and remember the child is dyslexic.
                If the retrieved context is relevant, try to incorporate its theme or sentiment subtly.
                '''
            )
            try:
                response = genai_model_therapy.generate_content(prompt)
                render_chatbot_html_with_text(response.text)
                # Optionally clear the query after processing
                # st.session_state.th_query_text = ""
            except Exception as e:
                st.error(f"Failed to generate response from GenAI for therapy: {e}")
    # else:
    #    st.warning("Please share your thoughts or use the microphone.")

# --- Main Application Logic ---
def main():
    st.title("🧩 Eduthon Dyslexia Support Hub 🧩")

    tab1, tab2, tab3 = st.tabs(["📖 Read Aloud", "💬 General Chatbot", "❤️‍🩹 Therapy Support"])

    with tab1:
        read_aloud_tab()
    
    with tab2:
        general_chatbot_tab()

    with tab3:
        therapy_chatbot_tab()

if __name__ == "__main__":
    main() 