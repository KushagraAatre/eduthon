# --- START OF FILE standalone.py (Modified Sections) ---

import streamlit as st
import streamlit.components.v1 as components
import easyocr
# PIL is still needed for other parts
from PIL import Image, UnidentifiedImageError
import fitz  # PyMuPDF
import io
from typing import Optional, List, Tuple, Any
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import google.generativeai as genai
import speech_recognition as sr
# import pyttsx3
import time
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration
# import nltk

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Eduthon Dyslexia Hub",
                   layout="wide", initial_sidebar_state="expanded")

# --- Inject Custom CSS for Lexend font for the entire Streamlit app ---
# (Keep your existing CSS injection here)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="st-"], .stApp, .stButton>button, .stTextArea textarea, .stTextInput input, .stFileUploader label, .stSelectbox div[data-baseweb="select"] > div, .stAlert, .stMarkdown, .stExpander header, h1, h2, h3, h4, h5, h6, p, div, span, li, label, button, input, select, textarea {
        font-family: 'Lexend', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Global Initializations & Helper Functions ---

# Initialize EasyOCR reader (still needed for PDF and Read Aloud OCR)
@st.cache_resource
def load_ocr_reader() -> easyocr.Reader:
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize EasyOCR reader: {e}")
        st.stop()


ocr_reader_instance = load_ocr_reader()

# Configure Google GenAI
try:
    # !!! IMPORTANT: Use Streamlit Secrets for API Keys !!!
    # Create a file .streamlit/secrets.toml with:
    # GENAI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
    GENAI_API_KEY = "AIzaSyD58wne48dBDpHbps2RQg_3rH08zLLHe_0"
    if not GENAI_API_KEY:
        st.error(
            "GENAI_API_KEY not found in Streamlit secrets. Chatbot features will be disabled.")
        genai_model_chat = None
        genai_model_therapy = None
        genai_model_vision = None  # For handwriting aid
    else:
        genai.configure(api_key=GENAI_API_KEY)
        # Model for general chat and therapy (can be text-only or multimodal)
        # Using gemini-1.5-flash-latest as it's versatile
        genai_model_chat = genai.GenerativeModel('gemini-2.0-flash')
        genai_model_therapy = genai.GenerativeModel('gemini-2.0-flash')
        # Explicitly define a model for vision tasks if you prefer,
        # or ensure genai_model_chat is vision-capable.
        # For this example, we'll use genai_model_chat if it's gemini-1.5-flash-latest.
        # Assuming gemini-1.5-flash-latest handles vision
        genai_model_vision = genai_model_chat
except Exception as e:
    st.error(
        f"Failed to configure Google GenAI: {e}. Chatbot features may not work.")
    genai_model_chat = None
    genai_model_therapy = None
    genai_model_vision = None

# ... (keep other helper functions like get_speech_recognizer, perform_ocr, extract_text_from_pdf, js_string_escape_ra, etc., as they are used by other tabs) ...
# --- Helper Functions from readaloud1.py (and others) ---


def perform_ocr(pil_image: Image.Image) -> str:
    extracted_text = ""
    try:
        img_byte_arr = io.BytesIO()
        # Default to PNG if format is None
        image_format = pil_image.format if pil_image.format else 'PNG'
        pil_image.save(img_byte_arr, format=image_format)
        img_bytes = img_byte_arr.getvalue()
        result: List[Tuple[Any, str, Any]
                     ] = ocr_reader_instance.readtext(img_bytes)
        if result:
            extracted_text = ' '.join([text[1] for text in result])
    except Exception as e:
        print(f"OCR Error: {e}")  # Log for server-side debugging
        st.error(f"An error occurred during OCR processing: {e}")
    return extracted_text.strip()


def extract_text_from_pdf(pdf_file_bytes: bytes) -> str:
    full_extracted_text: List[str] = []
    MIN_CHARS_FOR_DIRECT_TEXT = 50
    OCR_DPI = 150  # Increased DPI for potentially better OCR from PDF images
    try:
        pdf_document = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        num_pages = len(pdf_document)

        progress_bar_container = st.empty()  # Container for progress bar and text

        for page_num in range(num_pages):
            current_progress = (page_num + 1) / num_pages
            with progress_bar_container.container():  # Recreate progress bar in container to update text
                st.text(f"Processing PDF Page {page_num + 1}/{num_pages}...")
                st.progress(current_progress)

            page_text_content = ""
            try:
                page = pdf_document.load_page(page_num)
                direct_text = page.get_text("text").strip()
                ocr_text = ""

                # Heuristic: if direct text is very short, or seems like scanned image, try OCR
                if len(direct_text) < MIN_CHARS_FOR_DIRECT_TEXT or not any(c.isalnum() for c in direct_text):
                    with progress_bar_container.container():
                        st.text(
                            f"Page {page_num + 1}/{num_pages} (low text/image suspected, trying OCR)...")
                        st.progress(current_progress)
                    # Pass PIL image to perform_ocr
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    img_data = pix.tobytes("png")  # Get image data as bytes
                    pil_image_from_pdf = Image.open(io.BytesIO(img_data))
                    ocr_text = perform_ocr(pil_image_from_pdf).strip()

                # Prioritize OCR if it yields significantly more text than very short direct text
                if ocr_text and len(direct_text) < MIN_CHARS_FOR_DIRECT_TEXT and len(ocr_text) > len(direct_text):
                    page_text_content = ocr_text
                elif direct_text:
                    page_text_content = direct_text
                elif ocr_text:  # Fallback to OCR if direct text was empty
                    page_text_content = ocr_text

                if page_text_content:
                    full_extracted_text.append(page_text_content)
            except Exception as e_page:
                st.warning(
                    f"Could not process page {page_num + 1} of PDF: {e_page}")

        progress_bar_container.empty()  # Clear progress bar area
        st.success("PDF processing complete.")
        pdf_document.close()

    except fitz.errors.FitzError as fe:
        st.error(
            f"PyMuPDF Error: {fe}. The PDF might be corrupted or password-protected.")
        if 'progress_bar_container' in locals():
            progress_bar_container.empty()
    except Exception as e_doc:
        st.error(f"General error processing PDF document: {e_doc}")
        if 'progress_bar_container' in locals():
            progress_bar_container.empty()
    return "\n\n<page_break>\n\n".join(full_extracted_text).strip()

# ... (Keep other unchanged helper functions) ...


def js_string_escape_ra(s: str) -> str:
    if not s:
        return ""
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
        html_code = html_template.replace(
            '{{ text_content }}', escaped_text_content)
        # Pass the actual Unsplash key
        html_code = html_code.replace('{{ access_key }}', access_key)
        # Assuming nouns_for_images_json is not used in this version of readaloud1.html
        # If it is, you'd need to pass it and handle it
        html_code = html_code.replace(
            '{{ nouns_for_images_json }}', "[]")  # Default to empty list
        return html_code
    except FileNotFoundError:
        st.error("Error: 'readaloud1.html' not found.")
        return "<p><b>Error: TTS Player template (readaloud1.html) missing.</b></p>"
    except Exception as e:
        st.error(f"Error reading or processing 'readaloud1.html': {e}")
        return f"<p><b>Error loading TTS Player: {e}</b></p>"


def display_readaloud_tts_player(text_content: str):
    # !!! IMPORTANT: Use Streamlit Secrets for API Keys !!!
    # Fallback to empty if not found
    unsplash_key = st.secrets.get("UNSPLASH_ACCESS_KEY", "")
    html_content = get_readaloud_html_content(text_content, unsplash_key)
    components.html(html_content, height=600, scrolling=True)

# --- Helper Functions from chatbot.py (or adapted) ---


def recognize_speech_from_mic_ch() -> str:
    # Ensure get_speech_recognizer is defined and returns speech_r
    speech_r = get_speech_recognizer()  # Make sure speech_r is initialized
    with sr.Microphone() as source:
        try:
            speech_r.adjust_for_ambient_noise(source, duration=0.2)
            st.info("Listening...")
            audio = speech_r.listen(
                source, timeout=5, phrase_time_limit=10)  # Added timeout
            my_text = speech_r.recognize_google(audio)
            my_text = my_text.lower()
            st.success(f"Recognized: {my_text}")
            return my_text
        except sr.WaitTimeoutError:
            st.warning(
                "No speech detected within the time limit. Please try again.")
        except sr.RequestError as e:
            st.error(
                f"Could not request results from Google Speech Recognition service; {e}")
        except sr.UnknownValueError:
            st.warning(
                "Google Speech Recognition could not understand audio. Please try again.")
        except Exception as e:
            st.error(
                f"An unexpected error occurred during speech recognition: {e}")
        return ""


@st.cache_resource
def get_speech_recognizer():  # Definition was missing
    return sr.Recognizer()


def render_chatbot_html_with_text(text_variable: str):
    try:
        with open("chatbot.html", "r", encoding="utf-8") as file:
            html_code = file.read()
        escaped_text_variable = js_string_escape_ra(
            text_variable)
        html_code = html_code.replace("{text_variable}", escaped_text_variable)
        components.html(html_code, height=400, scrolling=True)
    except FileNotFoundError:
        st.error(
            "Error: 'chatbot.html' not found. Cannot display chatbot response visually.")
    except Exception as e:
        st.error(f"Error loading or processing 'chatbot.html': {e}")
# ... (Tab functions like read_aloud_tab, general_chatbot_tab, therapy_chatbot_tab remain largely the same,
#      except for the `understanding_aid_tab` which is modified below) ...

# --- Tab Functions ---


def read_aloud_tab():
    st.header("📖 Reading Assistance Pro (Text-to-Speech)")
    st.markdown(
        "Upload an image or PDF, or type text directly to have it read aloud.")

    # Initialize session state variables for this tab
    if "ra_manual_text_input" not in st.session_state:
        st.session_state.ra_manual_text_input = ""
    if "ra_image_extracted_text" not in st.session_state:
        st.session_state.ra_image_extracted_text = ""
    if "ra_pdf_extracted_text" not in st.session_state:
        st.session_state.ra_pdf_extracted_text = ""
    if "ra_last_uploaded_image_file_id" not in st.session_state:
        st.session_state.ra_last_uploaded_image_file_id = None
    if "ra_last_uploaded_pdf_file_id" not in st.session_state:
        st.session_state.ra_last_uploaded_pdf_file_id = None

    col1, col2 = st.columns([1, 2])  # Adjusted column ratio

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
                    st.session_state.ra_image_extracted_text = perform_ocr(
                        pil_image)
                    st.session_state.ra_manual_text_input = ""  # Clear other sources
                    st.session_state.ra_pdf_extracted_text = ""
                    st.session_state.ra_last_uploaded_image_file_id = uploaded_image_file.file_id
                    st.session_state.ra_last_uploaded_pdf_file_id = None
                    st.success("Image text extracted!")
                except UnidentifiedImageError:
                    st.error(
                        "The uploaded file is not a valid image or is corrupted.")
                except Exception as e:
                    st.error(f"Error processing image: {e}")

        st.markdown("**3. Upload a PDF**")
        uploaded_pdf_file = st.file_uploader(
            "Choose a PDF...", type=["pdf"], key="ra_pdf_uploader"
        )
        if uploaded_pdf_file is not None and uploaded_pdf_file.file_id != st.session_state.get("ra_last_uploaded_pdf_file_id"):
            st.session_state.ra_pdf_extracted_text = extract_text_from_pdf(
                uploaded_pdf_file.getvalue())
            st.session_state.ra_manual_text_input = ""  # Clear other sources
            st.session_state.ra_image_extracted_text = ""
            st.session_state.ra_last_uploaded_pdf_file_id = uploaded_pdf_file.file_id
            st.session_state.ra_last_uploaded_image_file_id = None
            # Success message is now inside extract_text_from_pdf

        if st.button("Clear All Inputs & Text", key="ra_clear_all"):
            st.session_state.ra_manual_text_input = ""
            st.session_state.ra_image_extracted_text = ""
            st.session_state.ra_pdf_extracted_text = ""
            st.session_state.ra_last_uploaded_image_file_id = None
            st.session_state.ra_last_uploaded_pdf_file_id = None
            st.success("All inputs and extracted text cleared.")
            st.rerun()

    with col2:
        st.subheader("Extracted Text & Player")
        texts_to_combine = []
        if st.session_state.ra_manual_text_input:
            texts_to_combine.append(
                f"--- MANUAL INPUT ---\n{st.session_state.ra_manual_text_input}")
        if st.session_state.ra_image_extracted_text:
            texts_to_combine.append(
                f"--- IMAGE TEXT ---\n{st.session_state.ra_image_extracted_text}")
        if st.session_state.ra_pdf_extracted_text:
            texts_to_combine.append(
                f"--- PDF TEXT ---\n{st.session_state.ra_pdf_extracted_text}")

        final_combined_text_for_tts = "\n\n".join(texts_to_combine).strip()

        if final_combined_text_for_tts:
            st.text_area("Final Text for Player:", value=final_combined_text_for_tts,
                         height=250, disabled=True, key="ra_final_text_ro")
            st.download_button(
                label="📥 Download Combined Text (.txt)", data=final_combined_text_for_tts.encode('utf-8'),
                file_name="readaloud_extracted_text.txt", mime="text/plain", key="ra_download_btn"
            )
            display_readaloud_tts_player(final_combined_text_for_tts)
        else:
            st.info(
                "No text provided or extracted yet. Use the input methods on the left to get started.")


def general_chatbot_tab():
    st.header("💬 General Chatbot for Dyslexic Individuals")
    st.markdown(
        "Ask questions and get simplified explanations. Type or use the microphone.")

    if genai_model_chat is None:
        st.error(
            "General Chatbot is unavailable: GenAI model not initialized. Check API Key.")
        return

    if 'gc_query_text' not in st.session_state:
        st.session_state.gc_query_text = ""
    if 'gc_response_text' not in st.session_state:
        st.session_state.gc_response_text = ""  # To store response

    user_query_input = st.text_input(
        "Type your query here:",
        value=st.session_state.gc_query_text,
        key="gc_text_input_widget",
        on_change=lambda: setattr(st.session_state, 'gc_query_text',
                                  st.session_state.gc_text_input_widget)  # Update state on change
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submit_text_btn = st.button(
            "Submit Text Query", key="gc_submit_text", use_container_width=True)
    with col_btn2:
        mic_btn = st.button("🎤 Use Microphone",
                            key="gc_start_mic", use_container_width=True)

    query_to_process = ""
    if submit_text_btn and st.session_state.gc_query_text.strip():
        query_to_process = st.session_state.gc_query_text.strip()

    if mic_btn:
        recognized_text = recognize_speech_from_mic_ch()
        if recognized_text:
            st.session_state.gc_query_text = recognized_text  # Update text box
            query_to_process = recognized_text
            st.rerun()  # Rerun to reflect mic input in text box and trigger processing if needed

    if query_to_process:
        st.info(f"Processing query: {query_to_process}")
        with st.spinner("Generating response..."):
            prompt = (
                f"User query: \"{query_to_process}\"\n\n"
                f"CONTEXT: You are an educational assistant AI designed to help a child or young person with dyslexia. "
                f"Your primary goal is to explain concepts and answer questions in a way that is easy for them to understand. "
                f"Please adhere to the following guidelines strictly:\n"
                f"1. Simple Language: Use short, clear sentences. Avoid jargon, complex vocabulary, and idiomatic expressions that might be confusing. Define any necessary specific terms very simply.\n"
                f"2. Chunking: Break down information into small, digestible chunks. Use bullet points or numbered lists if it helps clarify steps or multiple ideas.\n"
                f"3. Direct Answers: Get straight to the point. Avoid overly verbose or abstract explanations.\n"
                f"4. Visual Analogy (Descriptive): While you can't show images, you can use simple, concrete analogies if they aid understanding (e.g., 'think of it like...').\n"
                f"5. Positive and Encouraging Tone: Be supportive and patient in your language.\n"
                f"6. Repetition (if needed for clarity): It's okay to subtly rephrase key points if it reinforces understanding.\n"
                f"7. Focus: Stick to answering the user's query directly.\n"
                f"8. Brevity: Keep responses concise, ideally around 100-150 words, unless more detail is explicitly needed for clarity on a complex topic. Prefer shorter if possible.\n\n"
                f"RESPONSE:"
            )
            try:
                response = genai_model_chat.generate_content(prompt)
                st.session_state.gc_response_text = response.text
            except Exception as e:
                st.error(f"Failed to generate response from GenAI: {e}")
                st.session_state.gc_response_text = "Sorry, I couldn't process that request."

    if st.session_state.gc_response_text:
        st.subheader("Chatbot Response:")
        render_chatbot_html_with_text(st.session_state.gc_response_text)
        if st.button("Clear Chat", key="gc_clear_chat"):
            st.session_state.gc_query_text = ""
            st.session_state.gc_response_text = ""
            st.rerun()


@st.cache_resource
def load_therapy_faiss_index():
    try:
        df = pd.read_csv('dataset.csv', encoding='latin1')
        df['Query'] = df['Query'].astype(str).str.lower()
        df['Response'] = df['Response'].astype(
            str)  # Keep original case for display

        if df.empty or 'Query' not in df.columns or 'Response' not in df.columns:
            st.warning(
                "Therapy Chatbot: 'dataset.csv' is empty or missing 'Query'/'Response' columns. Retrieval will be disabled.")
            return None, None, None

        vectorizer = TfidfVectorizer()
        # Use lowercased queries for vectorization
        query_vectors = vectorizer.fit_transform(
            df['Query'])  # df['Query'] is already lowercased
        dimension = query_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        query_vectors_faiss = query_vectors.toarray().astype('float32')
        index.add(query_vectors_faiss)
        return index, vectorizer, df
    except FileNotFoundError:
        st.error(
            "Therapy Chatbot Error: 'dataset.csv' not found. Retrieval functionality will be limited.")
        return None, None, None
    except Exception as e:
        st.error(f"Error initializing Therapy Chatbot FAISS index: {e}")
        return None, None, None


therapy_faiss_index, therapy_vectorizer, therapy_df_full = load_therapy_faiss_index()


def therapy_chatbot_tab():
    st.header("❤️‍🩹 Therapy Support Chatbot")
    st.markdown(
        "A supportive space to discuss feelings and find encouragement. Type or use the microphone.")

    if genai_model_therapy is None:
        st.error(
            "Therapy Chatbot is unavailable: GenAI model not initialized. Check API Key.")
        return

    if therapy_faiss_index is None:
        st.warning("Therapy Chatbot retrieval features are limited due to an issue with 'dataset.csv'. It will rely solely on general AI responses.")

    if 'th_query_text' not in st.session_state:
        st.session_state.th_query_text = ""
    if 'th_response_text' not in st.session_state:
        st.session_state.th_response_text = ""

    user_query_input = st.text_input(
        "Share your thoughts or questions here:",
        value=st.session_state.th_query_text,
        key="th_text_input_widget",
        on_change=lambda: setattr(
            st.session_state, 'th_query_text', st.session_state.th_text_input_widget)
    )

    col_btn1_th, col_btn2_th = st.columns(2)
    with col_btn1_th:
        submit_text_btn_th = st.button(
            "Share Thoughts", key="th_submit_text", use_container_width=True)
    with col_btn2_th:
        mic_btn_th = st.button("🎤 Share via Microphone",
                               key="th_start_mic", use_container_width=True)

    query_to_process_th = ""
    if submit_text_btn_th and st.session_state.th_query_text.strip():
        query_to_process_th = st.session_state.th_query_text.strip()

    if mic_btn_th:
        recognized_text_th = recognize_speech_from_mic_ch()
        if recognized_text_th:
            st.session_state.th_query_text = recognized_text_th
            query_to_process_th = recognized_text_th
            st.rerun()

    if query_to_process_th:
        st.info(f"Processing: {query_to_process_th}")
        retrieved_context = ""
        if therapy_faiss_index and therapy_vectorizer and therapy_df_full is not None:
            try:
                query_vec = therapy_vectorizer.transform(
                    [str(query_to_process_th).lower()]).toarray().astype('float32')
                D, I = therapy_faiss_index.search(query_vec, k=1)
                # Check index bounds
                if I.size > 0 and 0 <= I[0][0] < len(therapy_df_full):
                    retrieved_context = therapy_df_full.iloc[I[0]
                                                             [0]]['Response']
                    st.caption(
                        f"Found related past interaction theme: {retrieved_context[:100]}...")
                else:  # Add this else block for clarity
                    st.caption(
                        "No closely matching past interaction found in dataset.")
            except Exception as e:
                st.warning(f"Could not retrieve context from dataset: {e}")

        with st.spinner("Thinking and preparing a supportive response..."):
            prompt = (
                f"User (who may be feeling vulnerable, possibly related to dyslexia or learning challenges): \"{query_to_process_th}\"\n\n"
                f"Similar past interaction's core message (for thematic context, if available): \"{retrieved_context}\"\n\n"
                f"INSTRUCTIONS FOR THERAPIST AI:\n"
                f"You are an AI therapist assistant. Your goal is to provide empathetic, supportive, and encouraging responses to a user, who might be a child or young person dealing with challenges, potentially including dyslexia. "
                f"Adhere to these principles:\n"
                f"1.  **Empathetic & Validating:** Acknowledge and validate the user's feelings (e.g., \"It sounds like you're feeling [emotion]...\", \"It's understandable to feel that way when...\").\n"
                f"2.  **Simple & Clear Language:** Use easy-to-understand words and short sentences, suitable for someone who may find reading difficult.\n"
                f"3.  **Positive Reframe (where appropriate):** If the user expresses negativity about themselves (e.g., related to dyslexia), gently reframe it. Emphasize dyslexia as a different way of learning, not a lack of intelligence. Highlight strengths associated with dyslexic thinking if relevant (creativity, problem-solving).\n"
                f"4.  **Encouragement & Hope:** Offer words of encouragement. Remind them of their potential and that challenges can be overcome.\n"
                f"5.  **Focus on Strengths:** Help them see their strengths.\n"
                f"6.  **Suggest Coping & Resources (General):** Briefly mention general strategies like talking to a trusted adult, using tools that help them, or breaking tasks into smaller steps. Avoid giving specific medical or diagnostic advice.\n"
                f"7.  **Successful Examples (Optional & Brief):** If appropriate and fits naturally, you can briefly mention that many successful people have dyslexia to inspire hope.\n"
                f"8.  **Conversational & Gentle Tone:** Be warm, approachable, and patient.\n"
                f"9.  **Brevity & Focus:** Keep responses concise (around 120-180 words). Focus on the user's immediate statement.\n"
                f"10. **Contextual Relevance:** If retrieved context was provided, subtly weave its theme or sentiment into your response if it aligns, but prioritize addressing the user's current input directly.\n\n"
                f"RESPONSE (as empathetic AI therapist):"
            )
            try:
                response = genai_model_therapy.generate_content(prompt)
                st.session_state.th_response_text = response.text
            except Exception as e:
                st.error(
                    f"Failed to generate response from GenAI for therapy: {e}")
                st.session_state.th_response_text = "I'm sorry, I'm having a little trouble responding right now. Please know your feelings are valid."

    if st.session_state.th_response_text:
        st.subheader("Supportive Response:")
        render_chatbot_html_with_text(st.session_state.th_response_text)
        if st.button("Clear Therapy Chat", key="th_clear_chat"):
            st.session_state.th_query_text = ""
            st.session_state.th_response_text = ""
            st.rerun()


def understanding_aid_tab():
    st.header("✍️ Understanding Aid for Handwriting (Direct Image Interpretation)")
    st.markdown(
        "Upload an image of handwriting, and the AI will attempt to interpret it directly from the image.")

    # Use genai_model_vision which should be a multimodal model like gemini-1.5-flash-latest
    if genai_model_vision is None:
        st.error(
            "Understanding Aid is unavailable: Vision AI model not initialized. Check API Key and model configuration.")
        return

    # Initialize session state variables for this tab
    # REMOVED: ua_ocr_text
    if "ua_interpreted_text" not in st.session_state:
        st.session_state.ua_interpreted_text = ""
    if "ua_last_uploaded_file_id" not in st.session_state:
        st.session_state.ua_last_uploaded_file_id = None
    if "ua_uploaded_image_bytes" not in st.session_state:
        st.session_state.ua_uploaded_image_bytes = None
    if "ua_uploaded_image_mime_type" not in st.session_state:  # ADDED for MIME type
        st.session_state.ua_uploaded_image_mime_type = None

    uploaded_hw_image = st.file_uploader(
        "Upload an image of handwriting (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"],  # Common image types
        key="ua_image_uploader"
    )

    if uploaded_hw_image is not None:
        if uploaded_hw_image.file_id != st.session_state.get("ua_last_uploaded_file_id"):
            st.session_state.ua_interpreted_text = ""  # Clear previous interpretation
            st.session_state.ua_last_uploaded_file_id = uploaded_hw_image.file_id
            st.session_state.ua_uploaded_image_bytes = uploaded_hw_image.getvalue()
            st.session_state.ua_uploaded_image_mime_type = uploaded_hw_image.type  # Store MIME type
            st.success("Image uploaded successfully. Ready for interpretation.")

    if st.session_state.ua_uploaded_image_bytes:
        st.subheader("Uploaded Handwriting Image")
        st.image(st.session_state.ua_uploaded_image_bytes,
                 use_column_width=True)

        if st.button("🤖 Help Me Understand This Handwriting (from Image)", key="ua_interpret_image_button", use_container_width=True):
            with st.spinner("AI is attempting to interpret the handwriting directly from the image... This may take a moment."):

                image_part = {
                    "mime_type": st.session_state.ua_uploaded_image_mime_type,
                    "data": st.session_state.ua_uploaded_image_bytes
                }

                # Adjusted prompt for direct image interpretation
                prompt_text = (
                    f"The following image contains handwriting. The writer may be a student with dyslexia or other learning differences, "
                    f"so the handwriting might exhibit characteristics like letter reversals (b/d, p/q), transpositions (was/saw), phonetic spellings, "
                    f"inconsistent spacing, omitted letters, or unusual letter formations.\n\n"
                    f"Your task is to carefully interpret the handwriting directly from this image and provide a corrected, more conventionally spelled, and grammatically coherent version "
                    f"of what the student was most likely trying to write. Focus on discerning the intended meaning and content. "
                    f"If some parts are highly ambiguous or illegible even after considering these factors, you can state that or offer the most plausible interpretation, perhaps noting the uncertainty.\n\n"
                    f"Teacher-Friendly Interpreted Version (Present as if explaining to a teacher what the student wrote. Be clear and direct. Output only the interpreted text, no preamble like 'Here is the interpretation'):"
                )

                try:
                    # Ensure genai_model_vision (or genai_model_chat if it's the same multimodal model) is used
                    response = genai_model_vision.generate_content(
                        [prompt_text, image_part])
                    st.session_state.ua_interpreted_text = response.text
                    st.success("AI interpretation from image complete!")
                except Exception as e:
                    st.error(f"Failed to get interpretation from AI: {e}")
                    st.session_state.ua_interpreted_text = "Sorry, an error occurred while trying to interpret the handwriting from the image."
                    # More detailed error for debugging
                    st.error(f"Details: {type(e).__name__} - {str(e)}")
                    if hasattr(response, 'prompt_feedback'):
                        st.warning(
                            f"Prompt Feedback: {response.prompt_feedback}")

    if st.session_state.ua_interpreted_text:
        st.subheader("AI's Interpretation of the Handwriting (from Image)")
        st.markdown(st.session_state.ua_interpreted_text)

    if st.session_state.ua_uploaded_image_bytes and st.button("Clear Handwriting Aid", key="ua_clear_all", use_container_width=True):
        st.session_state.ua_interpreted_text = ""
        st.session_state.ua_last_uploaded_file_id = None
        st.session_state.ua_uploaded_image_bytes = None
        st.session_state.ua_uploaded_image_mime_type = None  # Clear MIME type
        st.success("Handwriting aid inputs and results cleared.")
        st.rerun()


# --- Main Application Logic ---
def main():
    st.title("🧩 Eduthon Dyslexia Support Hub 🧩")
    st.markdown("---")

    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Read Aloud"

    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Read Aloud",
        "💬 General Chatbot",
        "❤️‍🩹 Therapy Support",
        "✍️ Understanding Aid"
    ])

    with tab1:
        if st.session_state.active_tab != "Read Aloud":
            st.session_state.active_tab = "Read Aloud"
        read_aloud_tab()
    with tab2:
        if st.session_state.active_tab != "General Chatbot":
            st.session_state.active_tab = "General Chatbot"
        general_chatbot_tab()
    with tab3:
        if st.session_state.active_tab != "Therapy Support":
            st.session_state.active_tab = "Therapy Support"
        therapy_chatbot_tab()
    with tab4:
        if st.session_state.active_tab != "Understanding Aid":
            st.session_state.active_tab = "Understanding Aid"
        understanding_aid_tab()


if __name__ == "__main__":
    main()

# --- END OF FILE standalone.py ---
