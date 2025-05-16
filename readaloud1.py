import streamlit as st
import easyocr
from PIL import Image, UnidentifiedImageError
import fitz  # PyMuPDF
import io
from typing import Optional, List, Tuple, Any

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Reading Assistance Pro", layout="wide", initial_sidebar_state="expanded")

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
# --- End Custom CSS Injection ---

# --- Configuration & Initialization ---

# Initialize the OCR reader (can be slow, so do it once globally)
# Using gpu=False for broader compatibility if no CUDA/GPU setup for EasyOCR.
# If you have a compatible GPU and CUDA installed, you can try gpu=True.
@st.cache_resource # Cache the reader resource across sessions/reruns
def load_ocr_reader() -> easyocr.Reader:
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize EasyOCR reader: {e}")
        st.error("The application cannot function without OCR. Please check your EasyOCR installation and dependencies.")
        st.stop() # Stop the app if OCR can't be initialized

reader = load_ocr_reader()

# --- Core Functions ---

def perform_ocr(pil_image: Image.Image) -> str:
    """
    Performs OCR on a PIL Image object.
    Returns extracted text or an empty string if an error occurs or no text is found.
    """
    extracted_text = ""
    try:
        # Convert PIL image to bytes for easyocr, as it's often more robust
        img_byte_arr = io.BytesIO()
        # Use the image's original format if available, else default to PNG
        image_format = pil_image.format if pil_image.format else 'PNG'
        pil_image.save(img_byte_arr, format=image_format)
        img_bytes = img_byte_arr.getvalue()

        result: List[Tuple[Any, str, Any]] = reader.readtext(img_bytes)
        if result:
            extracted_text = ' '.join([text[1] for text in result])
    except Exception as e:
        # Log to console for debugging, show user-friendly error
        print(f"OCR Error: {e}")
        st.error(f"An error occurred during OCR processing. The image might be unusual or too complex.")
    return extracted_text.strip()

def extract_text_from_pdf(pdf_file_bytes: bytes) -> str:
    """
    Extracts text from a PDF. Attempts direct text extraction first.
    If a page has little or no native text (heuristic), it renders the page as an image and performs OCR.
    """
    full_extracted_text: List[str] = []
    MIN_CHARS_FOR_DIRECT_TEXT = 50  # Heuristic: if direct text is less, try OCR
    OCR_DPI = 150  # DPI for rendering PDF pages to images for OCR

    try:
        pdf_document = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        num_pages = len(pdf_document)
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for page_num in range(num_pages):
            status_text.text(f"Processing PDF Page {page_num + 1}/{num_pages}...")
            page_text_content = ""
            try:
                page = pdf_document.load_page(page_num)
                
                # 1. Attempt direct text extraction
                direct_text = page.get_text("text").strip()
                
                # 2. If direct text is short, attempt OCR
                ocr_text = ""
                if len(direct_text) < MIN_CHARS_FOR_DIRECT_TEXT:
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    img_bytes = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    ocr_text = perform_ocr(pil_image).strip()
                
                # 3. Combine results for the page
                # Prefer OCR if it's significantly longer and direct text was short.
                if ocr_text and len(direct_text) < MIN_CHARS_FOR_DIRECT_TEXT and len(ocr_text) > len(direct_text):
                    page_text_content = ocr_text
                elif direct_text:
                    page_text_content = direct_text
                elif ocr_text: # Only OCR had text
                    page_text_content = ocr_text
                
                if page_text_content:
                    full_extracted_text.append(page_text_content)

            except Exception as e_page:
                st.warning(f"Could not process page {page_num + 1} of PDF: {e_page}")
                continue 
            progress_bar.progress((page_num + 1) / num_pages)
        
        status_text.text("PDF processing complete.")
        pdf_document.close()

    except fitz.errors.FitzError as fe:
        st.error(f"PyMuPDF Error: {fe}. The PDF might be corrupted, password-protected, or in an unsupported format.")
    except Exception as e_doc:
        st.error(f"General error processing PDF document: {e_doc}")
    
    return "\n\n<page_break>\n\n".join(full_extracted_text).strip() # Use a distinct page separator

def js_string_escape(s: str) -> str:
    """Escapes a string to be safely embedded in a JavaScript string literal."""
    if not s: return ""
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\t", "\\t")
        # For embedding in HTML script tags, more care might be needed
        # .replace("<", "\\u003C").replace(">", "\\u003E").replace("&", "\\u0026")
    )

def get_html_content(text_content: str, access_key: str) -> str:
    """
    Reads HTML template, replaces placeholders with actual values.
    Ensures text_content is properly escaped for JavaScript.
    """
    try:
        with open('readaloud1.html', 'r', encoding='utf-8') as file:
            html_template = file.read()
        
        escaped_text_content = js_string_escape(text_content)
        # Assuming access_key is a simple string not needing complex escaping for its context
        html_code = html_template.replace('{{ text_content }}', escaped_text_content)
        html_code = html_code.replace('{{ access_key }}', access_key) # Ensure this placeholder exists in your HTML
        return html_code
    except FileNotFoundError:
        st.error("Error: 'readaloud1.html' not found. The Text-to-Speech player cannot be loaded.")
        return "<p><b>Error: TTS Player template (readaloud1.html) missing.</b></p>"
    except Exception as e:
        st.error(f"Error reading or processing 'readaloud1.html': {e}")
        return f"<p><b>Error loading TTS Player: {e}</b></p>"

def display_tts_player(text_content: str, access_key: str):
    """Generates and displays the HTML for the TTS player."""
    html_content = get_html_content(text_content, access_key)
    st.components.v1.html(html_content, height=600, scrolling=True)

# --- Main Application ---
def main():
    st.sidebar.title("📖 Reading Assistance Pro")
    st.sidebar.markdown("---")

    # Initialize session state variables for storing extracted text and tracking file uploads
    if "manual_text_input" not in st.session_state:
        st.session_state.manual_text_input = ""
    if "image_extracted_text" not in st.session_state:
        st.session_state.image_extracted_text = ""
    if "pdf_extracted_text" not in st.session_state:
        st.session_state.pdf_extracted_text = ""
    if "last_uploaded_image_id" not in st.session_state: # To track unique file uploads
        st.session_state.last_uploaded_image_id = None
    if "last_uploaded_pdf_id" not in st.session_state:
        st.session_state.last_uploaded_pdf_id = None

    # --- Sidebar for Inputs ---
    st.sidebar.header("Input Methods")

    # Option 1: Manual Text Input
    st.session_state.manual_text_input = st.sidebar.text_area(
        "**1. Type or Paste Text**",
        value=st.session_state.manual_text_input,
        height=150,
        key="manual_text_area_widget",
        help="Enter text directly here."
    )

    # Option 2: Image Upload
    st.sidebar.markdown("**2. Upload an Image (OCR)**")
    uploaded_image_file = st.sidebar.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="image_uploader_widget",
        help="Upload an image file (JPG, JPEG, PNG) for text extraction."
    )

    if uploaded_image_file is not None:
        # Process if it's a new file
        if uploaded_image_file.id != st.session_state.get("last_uploaded_image_id"):
            st.sidebar.info(f"Processing new image: {uploaded_image_file.name}")
            with st.spinner("Extracting text from image..."):
                try:
                    pil_image = Image.open(uploaded_image_file)
                    st.session_state.image_extracted_text = perform_ocr(pil_image)
                    st.session_state.pdf_extracted_text = "" # Clear PDF text
                    st.session_state.last_uploaded_image_id = uploaded_image_file.id
                    st.session_state.last_uploaded_pdf_id = None # Reset PDF tracking
                    if st.session_state.image_extracted_text:
                        st.sidebar.success("Image text extracted!")
                    else:
                        st.sidebar.warning("No text found in image, or OCR failed.")
                except UnidentifiedImageError:
                    st.sidebar.error("Cannot process: Uploaded file is not a valid image.")
                except Exception as e:
                    st.sidebar.error(f"Error processing image: {e}")
        elif not st.session_state.image_extracted_text and uploaded_image_file.id == st.session_state.get("last_uploaded_image_id"):
            # If same file is there but no text (e.g. after clearing), re-process.
            # This case is less common with current flow but good for robustness.
             st.sidebar.info(f"Re-processing image: {uploaded_image_file.name}")
             with st.spinner("Re-extracting text from image..."):
                try:
                    pil_image = Image.open(uploaded_image_file)
                    st.session_state.image_extracted_text = perform_ocr(pil_image)
                except Exception: pass # Errors handled in perform_ocr


    # Option 3: PDF Upload
    st.sidebar.markdown("**3. Upload a PDF**")
    uploaded_pdf_file = st.sidebar.file_uploader(
        "Choose a PDF...",
        type=["pdf"],
        key="pdf_uploader_widget",
        help="Upload a PDF file for text extraction. Scanned PDFs will also be processed using OCR."
    )

    if uploaded_pdf_file is not None:
        if uploaded_pdf_file.id != st.session_state.get("last_uploaded_pdf_id"):
            st.sidebar.info(f"Processing new PDF: {uploaded_pdf_file.name}")
            # PDF processing feedback (progress bar, status text) is handled within extract_text_from_pdf
            st.session_state.pdf_extracted_text = extract_text_from_pdf(uploaded_pdf_file.getvalue())
            st.session_state.image_extracted_text = "" # Clear image text
            st.session_state.last_uploaded_pdf_id = uploaded_pdf_file.id
            st.session_state.last_uploaded_image_id = None # Reset image tracking
            if st.session_state.pdf_extracted_text:
                st.sidebar.success("PDF text extracted!")
            else:
                st.sidebar.warning("No text found in PDF, or extraction failed.")
        elif not st.session_state.pdf_extracted_text and uploaded_pdf_file.id == st.session_state.get("last_uploaded_pdf_id"):
            st.sidebar.info(f"Re-processing PDF: {uploaded_pdf_file.name}")
            st.session_state.pdf_extracted_text = extract_text_from_pdf(uploaded_pdf_file.getvalue())


    st.sidebar.markdown("---")
    # Clear All Button
    if st.sidebar.button("Clear All Inputs & Extractions", key="clear_all_btn", help="Resets all text fields and clears file selections."):
        st.session_state.manual_text_input = ""
        st.session_state.image_extracted_text = ""
        st.session_state.pdf_extracted_text = ""
        st.session_state.last_uploaded_image_id = None
        st.session_state.last_uploaded_pdf_id = None
        # Reset file uploaders by clearing their keys from session_state which forces re-initialization on rerun
        # st.session_state.image_uploader_widget = None # This might not work as expected for file_uploader
        # st.session_state.pdf_uploader_widget = None
        st.experimental_rerun() # Simplest way to clear uploaders

    # --- Main Area for Displaying Text and TTS Player ---
    st.title("📝 Extracted Text & 🔊 Read Aloud")
    st.markdown("---")

    # Display individual extractions in expanders for clarity
    # These show the *current* state of extracted texts
    if st.session_state.manual_text_input:
        with st.expander("Manually Entered Text", expanded=False):
            st.text_area("Manual Input Content:", value=st.session_state.manual_text_input, height=150, disabled=True, key="disp_manual_text_area")
    
    if st.session_state.image_extracted_text:
        with st.expander("Text Extracted from Image", expanded=True):
            if uploaded_image_file and uploaded_image_file.id == st.session_state.last_uploaded_image_id:
                try: # Show preview of the image that was processed
                    st.image(Image.open(uploaded_image_file), caption=f"Source: {uploaded_image_file.name}", width=200)
                except Exception: pass
            st.text_area("Image OCR Content:", value=st.session_state.image_extracted_text, height=200, disabled=True, key="disp_image_text_area")

    if st.session_state.pdf_extracted_text:
        with st.expander("Text Extracted from PDF", expanded=True):
            if uploaded_pdf_file and uploaded_pdf_file.id == st.session_state.last_uploaded_pdf_id:
                 st.caption(f"Source: {uploaded_pdf_file.name}")
            st.text_area("PDF Content:", value=st.session_state.pdf_extracted_text, height=200, disabled=True, key="disp_pdf_text_area")

    # Combine all available text for the TTS player
    # Prepending source headers for clarity in the combined text
    texts_to_combine = []
    if st.session_state.manual_text_input:
        texts_to_combine.append(f"{st.session_state.manual_text_input}")
    if st.session_state.image_extracted_text: # Will be empty if PDF was uploaded after
        texts_to_combine.append(f"{st.session_state.image_extracted_text}")
    if st.session_state.pdf_extracted_text: # Will be empty if Image was uploaded after
         texts_to_combine.append(f"{st.session_state.pdf_extracted_text}")
    
    combined_text = "\n\n".join(texts_to_combine).strip()

    if combined_text:
        st.subheader("Combined Text for Text-to-Speech")
        st.text_area("Review or Edit Combined Text:", value=combined_text, height=300, key="final_combined_text_display")

        st.download_button(
            label="📥 Download Combined Text (.txt)",
            data=combined_text.encode('utf-8'), # Ensure proper encoding
            file_name="reading_assistance_extracted_text.txt",
            mime="text/plain",
            key="download_combined_btn"
        )
        st.markdown("---")
        st.subheader("🔊 Read Aloud Player")
        st.markdown("""
        The player below will use the "Combined Text" shown above.
        *Ensure `readaloud1.html` is present in the same directory as this script.*
        """)
        
        # !!! IMPORTANT: For deployed applications, use Streamlit Secrets for API keys !!!
        # Example: access_key = st.secrets.get("YOUR_TTS_ACCESS_KEY_NAME", "default_or_fallback_key")
        access_key = 'F4nLejAZww7_NC1DB8SF7pf0CKQLQhr9kBaZ0w9TISI' # Placeholder or actual key
        
        display_tts_player(combined_text, access_key)
    else:
        st.info("No text available. Please provide input using the options in the sidebar.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed with ❤️")

if __name__ == "__main__":
    main()