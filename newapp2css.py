import streamlit as st
import pandas as pd
import whisper
import os
import tempfile
import logging
import re
import json
from openai import OpenAI, OpenAIError

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
BASE_DIR = os.path.dirname(__file__)
DATA_FILES = {
    "patients": os.path.join(BASE_DIR, "patients.csv"),
    "providers": os.path.join(BASE_DIR, "providers.csv"),
    "visits": os.path.join(BASE_DIR, "visits.csv"),
    "diagnoses": os.path.join(BASE_DIR, "diagnoses.csv"),
    "procedures": os.path.join(BASE_DIR, "procedures.csv")
}

# Model configurations
WHISPER_MODEL_SIZE = "base"
OPENAI_MODEL = "gpt-4o" # Use GPT-4o as requested

# Target discharge summary sections based on user document
# Order matters for the LLM prompt structure
SUMMARY_SECTIONS_ORDERED = [
    "Patient Demographics",
    "Admission Details",
    "Discharge Diagnosis",
    "Procedures",
    "History of Present Illness",
    "Hospital Course",
    "Significant Findings",
    "Discharge Medications",
    "Condition at Discharge",
    "Discharge Instructions/Recommendations",
    "Follow-up",
    "Provider Information"
]

# --- Core Functions ---

# @st.cache_data # Re-enable caching if needed, but clear cache if data files change
def load_data(file_path):
    """Loads a single CSV file into a DataFrame, handling errors and stripping IDs."""
    try:
        df = pd.read_csv(file_path, dtype=str) # Read all as string initially
        # logging.info(f"Successfully loaded {file_path}") # Reduce verbosity
        # Strip whitespace from potential ID columns
        id_columns = ['PatientID', 'VisitID', 'ProviderID', 'DischargingPhysicianID',
                      'DiagnosingProviderID', 'PerformingPhysicianID', 'DiagnosisID', 'ProcedureID']
        for col in id_columns:
             if col in df.columns:
                df[col] = df[col].str.strip()
        # Convert date columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: Mock data file not found: {file_path}.")
        logging.error(f"Mock data file not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading {file_path}: {e}")
        logging.error(f"Error loading {file_path}: {e}")
        return None

# @st.cache_resource # Re-enable caching if needed
def load_whisper_model(model_size=WHISPER_MODEL_SIZE):
    """Loads the specified Whisper model."""
    try:
        logging.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        logging.info(f"Whisper model '{model_size}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model '{model_size}': {e}. Ensure FFmpeg is installed.")
        logging.error(f"Error loading Whisper model '{model_size}': {e}")
        return None

def transcribe_audio(model, audio_path):
    """Transcribes audio file using the loaded Whisper model."""
    if model is None: return "Error: Transcription model not loaded."
    try:
        logging.info(f"Starting transcription for: {audio_path}")
        if not os.path.exists(audio_path): return f"Error: Audio file not found at {audio_path}"
        result = model.transcribe(audio_path, fp16=False)
        logging.info(f"Transcription successful for: {audio_path}")
        return result["text"]
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        logging.error(f"Transcription failed for {audio_path}: {e}")
        return f"Transcription failed. Error: {e}"

def extract_patient_identifiers(text):
    """Basic extraction of patient name/ID using regex."""
    # ... (Function remains the same) ...
    extracted = {"name": None, "id": None}
    if not text: return extracted
    try:
        name_patterns = [
            r'(?:patient name is|patient is|for patient|summary for|note for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:patient id|id)'
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                extracted["name"] = name_match.group(1).strip()
                break
        id_patterns = [
             r'(?:patient id|id)\s+([A-Z]{3}\d+)',
             r'(?:patient id|id)\s+(\d+)'
        ]
        for pattern in id_patterns:
            id_match = re.search(pattern, text, re.IGNORECASE)
            if id_match:
                extracted["id"] = id_match.group(1).strip()
                break
    except Exception as e:
        logging.error(f"Error during identifier extraction: {e}")
    logging.info(f"Extracted identifiers from transcription: {extracted}")
    return extracted

def validate_transcription_vs_selection(transcription, selected_patient_id, selected_patient_name):
    """Compares extracted identifiers with selected patient data."""
    # ... (Function remains the same) ...
    if not transcription or not selected_patient_id or not selected_patient_name:
        return False, "Missing data for validation."

    extracted = extract_patient_identifiers(transcription)
    name_in_audio = extracted.get("name")
    id_in_audio = extracted.get("id")

    mismatch = False
    message = ""

    if name_in_audio or id_in_audio:
        logging.info(f"Validating Audio (Name: {name_in_audio}, ID: {id_in_audio}) vs Selection (Name: {selected_patient_name}, ID: {selected_patient_id})")
        if name_in_audio and selected_patient_name:
            if name_in_audio.lower().strip() not in selected_patient_name.lower().strip():
                mismatch = True
                message += f"Patient name in audio ('{name_in_audio}') does not match selected patient ('{selected_patient_name}'). "
                logging.warning(f"Name mismatch: Audio='{name_in_audio}', Selected='{selected_patient_name}'")

        if id_in_audio and selected_patient_id:
            audio_id_norm = id_in_audio.lower().strip()
            selected_id_norm = selected_patient_id.lower().strip()
            selected_id_num_only = selected_id_norm.replace("pat","")

            if audio_id_norm != selected_id_norm and audio_id_norm != selected_id_num_only:
                mismatch = True
                message += f"Patient ID in audio ('{id_in_audio}') does not match selected patient ID ('{selected_patient_id}')."
                logging.warning(f"ID mismatch: Audio='{id_in_audio}', Selected='{selected_patient_id}'")

    if mismatch:
        return False, f"Validation Error: {message.strip()} Please check the selected patient or the audio recording."
    else:
        return True, "Validation passed (or no conflicting identifiers found/matched in audio)."


def get_openai_api_key():
    """Gets the OpenAI API key from Streamlit secrets or environment variables."""
    # ... (Function remains the same) ...
    try:
        if hasattr(st, 'secrets') and "openai" in st.secrets and "api_key" in st.secrets["openai"]:
             return st.secrets["openai"]["api_key"]
    except Exception as e:
        logging.warning(f"Could not access Streamlit secrets (normal for local dev): {e}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    else:
        logging.warning("OpenAI API key not found in secrets or environment variable.")
        return None

def generate_summary_with_llm(structured_data, transcription):
    """Generates discharge summary using OpenAI API."""
    # ... (Function remains the same as previous version) ...
    api_key = get_openai_api_key()
    if not api_key:
        error_msg = "Error: API Key not configured."
        st.error("OpenAI API Key not found. Please set it via environment variable (OPENAI_API_KEY) or Streamlit secrets.")
        return error_msg

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        error_msg = f"Error: Failed to initialize OpenAI client - {e}"
        st.error(error_msg)
        logging.error(f"OpenAI client initialization failed: {e}")
        return error_msg

    structured_data_str = ""
    for key, value in structured_data.items():
        structured_data_str += f"**{key}:**\n"
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                structured_data_str += f"- {sub_key}: {sub_value}\n"
        elif isinstance(value, list) and value:
             structured_data_str += "\n".join([f"- {item}" for item in value]) + "\n"
        elif value:
             structured_data_str += f"- {value}\n"
        else:
             structured_data_str += f"- None Recorded\n"
        structured_data_str += "\n"

    section_list_str = "\n".join([f"- {section}" for section in SUMMARY_SECTIONS_ORDERED])

    system_prompt = f"""
You are an expert AI medical scribe creating a hospital discharge summary.
Your goal is to synthesize the provided Structured Data and the Nurse's Voice Note Transcription into a professional, accurate, and well-formatted discharge summary using Markdown.

**Output Format Requirements:**
- Use Markdown Level 3 Headings (`###`) for each section title EXACTLY as listed below.
- The section order MUST follow this specific sequence:
{section_list_str}
- Populate each section with relevant information synthesized from BOTH the Structured Data and the Nurse's Transcription. Prioritize structured data for factual elements (dates, demographics, recorded diagnoses/procedures, provider names). Use the transcription for narrative details (context, findings, medications, instructions, follow-up, condition).
- **Discharge Medications Section:** Format this section ONLY as a Markdown table with columns: `| Medication | Dosage | Frequency | Notes |`. Extract details carefully from the transcription. If details like dosage or frequency are missing in the transcription, leave the corresponding table cell blank or write 'Not specified'. If the transcription refers elsewhere (e.g., "see script"), state that clearly in the 'Notes' column or as a general note.
- **Handling Missing Information:** If NO relevant information can be found for a section (especially narrative ones) in EITHER the structured data OR the transcription, **INCLUDE the section header** (e.g., `### Follow-up`) but add the text `[No information provided in transcription or structured data for this section.]` below it. Do not omit headers.
- **Handling Conflicts/Vagueness:** If the transcription mentions information conflicting with structured data, include the transcription's version but add a note like `[Nurse Note: Verification Needed - structured data shows X]` or similar. If the transcription is vague (e.g., "follow up soon"), include the vague statement but add a note like `[Nurse Note: Specific timeframe/details needed]`.
- **Professional Tone:** Maintain a concise, professional, and medically appropriate tone. Avoid conversational filler unless clinically relevant.
- **Start Directly:** Begin the output directly with the first relevant section header (`### Patient Demographics`). Do not include introductory phrases.
"""

    user_prompt = f"""
Generate the discharge summary based on the following information, adhering strictly to the formatting and content instructions in the system prompt:

**Structured Data:**
{structured_data_str}

**Nurse's Voice Note Transcription:**
```
{transcription}
```
"""

    logging.info(f"Sending request to OpenAI model: {OPENAI_MODEL}")
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2500
        )
        summary = response.choices[0].message.content
        logging.info("Received response from OpenAI.")
        if not summary or not "###" in summary:
             logging.warning(f"LLM response might be invalid or empty: {summary[:200]}...")
             error_message = "Error: LLM generated an invalid or empty response. Please review transcription and try again, or edit manually."
             st.warning(error_message)
             return error_message
        return summary.strip()

    except OpenAIError as e:
        error_msg = f"Error: OpenAI API Error - {e}"
        st.error(error_msg)
        logging.error(f"OpenAI API Error: {e}")
        return error_msg
    except Exception as e:
        error_msg = f"Error: An unexpected error occurred calling OpenAI - {e}"
        st.error(error_msg)
        logging.error(f"Unexpected error calling OpenAI: {e}")
        return error_msg

# --- Helper Functions ---
def format_date_display(date_obj):
    """Formats datetime objects for display, handling NaT."""
    # ... (Function remains the same) ...
    if pd.isna(date_obj): return "N/A"
    try:
        return pd.to_datetime(date_obj).strftime('%Y-%m-%d')
    except: return "Invalid Date"

def get_display_provider_name(provider_id, providers_df):
    """Looks up provider name from ProviderID for display."""
    # ... (Function remains the same) ...
    if provider_id is None or pd.isna(provider_id) or providers_df.empty: return "N/A"
    provider = providers_df[providers_df['ProviderID'] == str(provider_id).strip()]
    return provider['ProviderName'].iloc[0] if not provider.empty else f"Unknown ({provider_id})"

# --- Main Streamlit App Logic ---

def main():
    """Main function to run the Streamlit application."""

    st.set_page_config(layout="wide", page_title="Discharge Summary AI", initial_sidebar_state="collapsed")

    # --- Inject Custom CSS ---
    st.markdown("""
    <style>
        /* Set a light grey background for the main app area */
        .stApp {
            background-color: #f0f2f6; /* Light grey background */
        }
        /* Add padding and subtle border to the main content block */
        .main .block-container {
            padding: 2rem 3rem 3rem 3rem; /* More padding */
            background-color: #ffffff; /* White background for content */
            border-radius: 10px; /* Rounded corners */
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Subtle shadow */
            border: 1px solid #e6e6e6; /* Light border */
        }
        /* Style headers */
        h1 {
            color: #005A9C; /* Darker blue for main title */
            border-bottom: 2px solid #005A9C;
            padding-bottom: 0.3em;
        }
        h2 {
            color: #0073B7; /* Slightly lighter blue for step headers */
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            border-bottom: 1px solid #d0d0d0;
            padding-bottom: 0.2em;
        }
         h3 {
            color: #333333; /* Dark grey for subheaders */
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        /* Style buttons */
        .stButton>button {
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            border: none; /* Remove default border */
            color: white;
            background-color: #0073B7; /* Match h2 color */
            transition: background-color 0.3s ease; /* Smooth hover effect */
        }
        .stButton>button:hover {
            background-color: #005A9C; /* Darker blue on hover */
        }
        .stButton>button:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        /* Style containers used for display */
        .stContainer {
            border: 1px solid #dcdcdc !important; /* Add !important to override potential defaults */
            border-radius: 8px !important;
            padding: 1rem !important;
        }
        /* Style captions */
        .stCaption {
            color: #555555;
            font-style: italic;
        }
        /* Ensure markdown tables look decent */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }
        th, td {
            border: 1px solid #dcdcdc;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f0f2f6; /* Light grey header */
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("👩‍⚕️ Discharge Summary AI Prototype")
    st.caption("Generate discharge summary drafts from voice notes and structured data.")
    # Removed the st.markdown("---") here as headers now have bottom borders

    # --- Load Data ---
    # ... (Data loading remains the same) ...
    all_data = {}
    data_loaded_successfully = True
    for name, path in DATA_FILES.items():
        df = load_data(path)
        if df is None:
            data_loaded_successfully = False
            break
        all_data[name] = df
    if not data_loaded_successfully:
        st.error("Application cannot start because one or more data files failed to load.")
        st.stop()
    patients_df = all_data["patients"]
    providers_df = all_data["providers"]
    visits_df = all_data["visits"]
    diagnoses_df = all_data["diagnoses"]
    procedures_df = all_data["procedures"]

    # --- Initialize Session State ---
    # ... (Session state initialization remains the same) ...
    if 'transcription' not in st.session_state: st.session_state.transcription = ""
    if 'llm_summary' not in st.session_state: st.session_state.llm_summary = ""
    if 'current_visit_id' not in st.session_state: st.session_state.current_visit_id = None
    if 'current_patient_id' not in st.session_state: st.session_state.current_patient_id = None
    if 'current_patient_name' not in st.session_state: st.session_state.current_patient_name = None
    if 'structured_data_for_llm' not in st.session_state: st.session_state.structured_data_for_llm = {}
    if 'validation_error' not in st.session_state: st.session_state.validation_error = None
    if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False

    # --- Step 1: Select Patient & Visit ---
    st.header("Step 1: Select Patient and Visit")
    sel_col1, sel_col2 = st.columns([1, 2])

    with sel_col1:
        # ... (Patient selection logic remains the same) ...
        st.subheader("Patient")
        try:
            patients_df_sorted = patients_df.sort_values(by='Name')
            patient_options = {f"{row['Name']} ({row['PatientID']})": row['PatientID'] for index, row in patients_df_sorted.iterrows()}
            _selected_patient_display = st.selectbox("", options=patient_options.keys(), key="patient_selector_widget", label_visibility="collapsed")
            _selected_patient_id = patient_options[_selected_patient_display]
            _selected_patient_name = _selected_patient_display.split(' (')[0]

            if st.session_state.current_patient_id != _selected_patient_id:
                 st.session_state.current_patient_id = _selected_patient_id
                 st.session_state.current_patient_name = _selected_patient_name
                 st.session_state.current_visit_id = None
                 st.session_state.transcription = ""
                 st.session_state.llm_summary = ""
                 st.session_state.structured_data_for_llm = {}
                 st.session_state.validation_error = None
                 st.session_state.processing_complete = False
                 logging.info(f"Patient selection changed to: {st.session_state.current_patient_id}")
                 st.rerun()
        except Exception as e:
            st.error(f"Error setting up patient selection: {e}")
            st.session_state.current_patient_id = None
            st.session_state.current_patient_name = None
            st.stop()

    with sel_col2:
        # ... (Visit selection and data prep logic remains the same) ...
        st.subheader("Visit")
        selected_visit_id = None
        if st.session_state.current_patient_id:
            try:
                patient_visits = visits_df[visits_df['PatientID'] == st.session_state.current_patient_id].sort_values(by='AdmissionDate', ascending=False)
            except Exception as e:
                 st.error(f"An error occurred filtering visits: {e}")
                 patient_visits = pd.DataFrame()

            if not patient_visits.empty:
                visit_options = { f"Visit {row['VisitID']} (Adm: {format_date_display(row['AdmissionDate'])} - Dis: {format_date_display(row['DischargeDate'])}) - Dx: {row.get('AdmissionDiagnosis', 'N/A')}": row['VisitID'] for index, row in patient_visits.iterrows()}
                current_visit_index = 0
                valid_visit_ids = list(visit_options.values())
                if st.session_state.current_visit_id in valid_visit_ids:
                     current_visit_index = valid_visit_ids.index(st.session_state.current_visit_id)
                else:
                     if st.session_state.current_visit_id is not None:
                         st.session_state.current_visit_id = None

                selected_visit_display = st.selectbox("", options=visit_options.keys(), key="visit_selector_widget", index=current_visit_index, label_visibility="collapsed")
                _selected_visit_id = visit_options[selected_visit_display]

                if st.session_state.current_visit_id != _selected_visit_id:
                    st.session_state.current_visit_id = _selected_visit_id
                    st.session_state.transcription = ""
                    st.session_state.llm_summary = ""
                    st.session_state.structured_data_for_llm = {}
                    st.session_state.validation_error = None
                    st.session_state.processing_complete = False
                    logging.info(f"Visit selection changed to: {st.session_state.current_visit_id}")
                    try:
                        patient_info = patients_df[patients_df['PatientID'] == st.session_state.current_patient_id].iloc[0]
                        visit_info = patient_visits[patient_visits['VisitID'] == st.session_state.current_visit_id].iloc[0]
                        visit_diagnoses = diagnoses_df[diagnoses_df['VisitID'] == st.session_state.current_visit_id]
                        visit_procedures = procedures_df[procedures_df['VisitID'] == st.session_state.current_visit_id]
                        discharging_physician_id = visit_info.get('DischargingPhysicianID', None)
                        discharging_physician_name = get_display_provider_name(discharging_physician_id, providers_df)
                        st.session_state.structured_data_for_llm = {
                            "Patient Demographics": {"Name": patient_info.get('Name', 'N/A'),"DOB": format_date_display(patient_info.get('DateOfBirth', '')),"Patient ID": st.session_state.current_patient_id,"Gender": patient_info.get('Gender', 'N/A')},
                            "Admission Details": {"Visit ID": st.session_state.current_visit_id,"Admission Date": format_date_display(visit_info.get('AdmissionDate', '')),"Discharge Date": format_date_display(visit_info.get('DischargeDate', '')),"Admission Diagnosis": visit_info.get('AdmissionDiagnosis', 'N/A'),"Department": visit_info.get('Department', 'N/A')},
                            "Provider Information": {"Discharging Physician": discharging_physician_name},
                            "Diagnoses Recorded": visit_diagnoses[['DiagnosisDescription', 'DiagnosisCode', 'DiagnosisType']].to_dict('records') if not visit_diagnoses.empty else [],
                            "Procedures Recorded": visit_procedures[['ProcedureName', 'ProcedureDate']].apply(lambda row: f"{row.get('ProcedureName','N/A')} ({format_date_display(row.get('ProcedureDate',''))})", axis=1).tolist() if not visit_procedures.empty else []
                        }
                        logging.info(f"Prepared structured data for LLM for Visit ID: {st.session_state.current_visit_id}")
                    except Exception as e:
                         st.error(f"An error occurred preparing structured data: {e}")
                         logging.error(f"Error during structured data prep: {e}")
                         st.session_state.structured_data_for_llm = {"error": "Failed to prepare structured data"}
                    st.rerun()
                elif not st.session_state.structured_data_for_llm: # If visit didn't change but data is missing
                     try:
                        patient_info = patients_df[patients_df['PatientID'] == st.session_state.current_patient_id].iloc[0]
                        visit_info = patient_visits[patient_visits['VisitID'] == st.session_state.current_visit_id].iloc[0]
                        visit_diagnoses = diagnoses_df[diagnoses_df['VisitID'] == st.session_state.current_visit_id]
                        visit_procedures = procedures_df[procedures_df['VisitID'] == st.session_state.current_visit_id]
                        discharging_physician_id = visit_info.get('DischargingPhysicianID', None)
                        discharging_physician_name = get_display_provider_name(discharging_physician_id, providers_df)
                        st.session_state.structured_data_for_llm = {
                            "Patient Demographics": {"Name": patient_info.get('Name', 'N/A'),"DOB": format_date_display(patient_info.get('DateOfBirth', '')),"Patient ID": st.session_state.current_patient_id,"Gender": patient_info.get('Gender', 'N/A')},
                            "Admission Details": {"Visit ID": st.session_state.current_visit_id,"Admission Date": format_date_display(visit_info.get('AdmissionDate', '')),"Discharge Date": format_date_display(visit_info.get('DischargeDate', '')),"Admission Diagnosis": visit_info.get('AdmissionDiagnosis', 'N/A'),"Department": visit_info.get('Department', 'N/A')},
                            "Provider Information": {"Discharging Physician": discharging_physician_name},
                            "Diagnoses Recorded": visit_diagnoses[['DiagnosisDescription', 'DiagnosisCode', 'DiagnosisType']].to_dict('records') if not visit_diagnoses.empty else [],
                            "Procedures Recorded": visit_procedures[['ProcedureName', 'ProcedureDate']].apply(lambda row: f"{row.get('ProcedureName','N/A')} ({format_date_display(row.get('ProcedureDate',''))})", axis=1).tolist() if not visit_procedures.empty else []
                        }
                        logging.info(f"Prepared structured data for LLM for Visit ID: {st.session_state.current_visit_id}")
                     except Exception as e:
                         st.error(f"An error occurred preparing structured data on initial load: {e}")
                         logging.error(f"Error during initial structured data prep: {e}")
                         st.session_state.structured_data_for_llm = {"error": "Failed to prepare structured data"}

            else: # No visits found
                st.warning(f"No visit data found for Patient ID: {st.session_state.current_patient_id}")
                if st.session_state.current_visit_id is not None:
                    st.session_state.current_visit_id = None
                    st.session_state.transcription = ""
                    st.session_state.llm_summary = ""
                    st.session_state.structured_data_for_llm = {}
                    st.session_state.validation_error = None
                    st.session_state.processing_complete = False
        else: # No patient selected
            st.info("Select a Patient to view Visits.")
            if st.session_state.current_visit_id is not None:
                 st.session_state.current_visit_id = None
                 st.session_state.transcription = ""
                 st.session_state.llm_summary = ""
                 st.session_state.structured_data_for_llm = {}
                 st.session_state.validation_error = None
                 st.session_state.processing_complete = False

    # Removed the st.markdown("---") here

    # --- Step 2: Upload Audio & Process ---
    st.header("Step 2: Upload Audio & Generate Summary")
    valid_visit_selected = st.session_state.current_visit_id is not None

    if valid_visit_selected:
        uploaded_audio = st.file_uploader("Upload Nurse Voice Note (.wav, .mp3, .m4a, .aac, etc.):", type=None, key="audio_uploader")

        # Display validation error from previous attempts if it exists
        if st.session_state.validation_error:
            st.error(st.session_state.validation_error)

        process_button_clicked = st.button("Transcribe and Generate Summary", key="process_button", disabled=(uploaded_audio is None))

        if uploaded_audio is not None:
            st.audio(uploaded_audio) # Show audio player only if file is uploaded

            if process_button_clicked:
                # Reset state for new processing run
                st.session_state.llm_summary = ""
                st.session_state.transcription = ""
                st.session_state.validation_error = None
                st.session_state.processing_complete = False

                # --- 1. Transcription ---
                # ... (Transcription logic remains the same) ...
                transcription_result = "Error: Could not process audio file."
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio_file:
                    tmp_audio_file.write(uploaded_audio.getvalue())
                    audio_path = tmp_audio_file.name
                logging.info(f"Temporary audio file saved at: {audio_path}")
                model = load_whisper_model()
                if model:
                    with st.spinner("Transcribing audio..."):
                        transcription_result = transcribe_audio(model, audio_path)
                else:
                    transcription_result = "Error: Transcription model failed to load."
                st.session_state.transcription = transcription_result # Store result
                try:
                    os.remove(audio_path)
                    logging.info(f"Temporary audio file removed: {audio_path}")
                except OSError as e:
                    logging.error(f"Error removing temporary file {audio_path}: {e}")

                # --- 2. Validation (only if transcription succeeded) ---
                if "error" not in transcription_result.lower():
                    st.success("Transcription complete!")
                    is_valid, validation_msg = validate_transcription_vs_selection(
                        st.session_state.transcription,
                        st.session_state.current_patient_id,
                        st.session_state.current_patient_name
                    )
                    logging.info(validation_msg)

                    if is_valid:
                        # --- 3. LLM Generation (if valid) ---
                        # ... (LLM call logic remains the same) ...
                        if st.session_state.structured_data_for_llm and "error" not in st.session_state.structured_data_for_llm:
                            with st.spinner("Generating discharge summary draft using AI..."):
                                llm_output = generate_summary_with_llm(
                                    st.session_state.structured_data_for_llm,
                                    st.session_state.transcription
                                )
                            st.session_state.llm_summary = llm_output
                            if "error" not in llm_output.lower():
                                 st.success("AI summary draft generated!")
                                 st.session_state.processing_complete = True
                        else:
                            st.error("Cannot generate summary: structured data preparation failed.")
                            st.session_state.llm_summary = "Error: Failed to prepare structured data."
                    else:
                        # Validation failed
                        st.session_state.validation_error = validation_msg
                        # Error will be displayed on rerun
                        st.session_state.llm_summary = ""
                else:
                    # Transcription failed
                    st.error(f"Transcription failed. Cannot generate summary. Details: {transcription_result}")
                    st.session_state.llm_summary = ""

                # Rerun needed to display results/errors
                st.rerun()

        # --- Display Transcription below button ---
        if st.session_state.transcription:
             # Use an expander for the transcription
             with st.expander("View Audio Transcription", expanded=False):
                 st.text_area("", value=st.session_state.get('transcription', ''), height=200, key="transcription_display_step2", disabled=True, label_visibility="collapsed")

    else: # No visit selected
        st.info("Please select a Patient and Visit before uploading audio.")

    # Removed the st.markdown("---") here

    # --- Step 3: Review & Edit ---
    st.header("Step 3: Review and Edit Summary")

    # Display AI Summary and Editable Area side-by-side
    edit_col1, edit_col2 = st.columns(2)

    with edit_col1:
        st.subheader("AI Generated Draft")
         # Determine what to display based on state
        display_summary = ""
        if st.session_state.validation_error:
            display_summary = f"**Validation Error:**\n\n{st.session_state.validation_error}\n\n*Summary generation skipped.*"
        elif not st.session_state.processing_complete and "error" in st.session_state.get('transcription','').lower():
             display_summary = f"*Transcription failed: {st.session_state.transcription}*"
        elif not st.session_state.processing_complete and not st.session_state.transcription:
             display_summary = '*Upload audio and click "Transcribe and Generate Summary" first.*'
        elif "error" in st.session_state.get('llm_summary','').lower():
             display_summary = f"*LLM generation failed:*\n\n{st.session_state.llm_summary}"
        else:
             display_summary = st.session_state.get('llm_summary', '*Summary will appear here after processing.*')

        # Display the AI output (or status message) using markdown
        summary_container = st.container(height=600, border=True)
        summary_container.markdown(display_summary, unsafe_allow_html=True)
        st.caption("This is the AI draft. Make final corrections in the area to the right.")


    with edit_col2:
        st.subheader("Final Editable Summary")
        # Pre-fill with LLM summary only if generation was successful
        edit_value = ""
        if st.session_state.processing_complete and not st.session_state.validation_error and "error" not in st.session_state.get('llm_summary', '').lower():
             edit_value = st.session_state.llm_summary

        final_summary_edit = st.text_area(
            "", # No label needed due to subheader
            value=edit_value,
            height=600,
            key="final_summary_edit_area",
            label_visibility="collapsed",
            # Disable if processing hasn't completed successfully
            disabled=not st.session_state.processing_complete or bool(st.session_state.validation_error) or "error" in st.session_state.get('llm_summary', '').lower()
        )
        st.caption("Make any final corrections or additions here before sending.")

    # Removed the st.markdown("---") here

    # --- Step 4: Finalize ---
    st.header("Step 4: Finalize")

    # --- Formatted Final Summary Preview ---
    st.subheader("Formatted Final Summary Preview")
    with st.expander("Click to view formatted preview based on edits", expanded=False):
        # Use the content from the editable text area for this preview
        preview_content = final_summary_edit if final_summary_edit else "*No summary content in edit area yet.*"
        preview_container = st.container(height=600, border=True) # Add container for consistency
        preview_container.markdown(preview_content, unsafe_allow_html=True)
    # --- END NEW ---

    st.markdown("---") # Add divider before button

    # Disable button if processing hasn't completed successfully or validation failed
    finalize_disabled = not st.session_state.processing_complete or bool(st.session_state.validation_error) or "error" in st.session_state.get('llm_summary', '').lower()

    if st.button("✅ Mark as Ready for Doctor Review", key="send_button", disabled=finalize_disabled, use_container_width=True):
        if valid_visit_selected:
            # In a real application, save `final_summary_edit` content
            st.success(f"Discharge summary for Visit {st.session_state.current_visit_id} marked as ready! (Simulation)")
            logging.info(f"Summary for Visit {st.session_state.current_visit_id} marked ready.")
            st.balloons()
        else:
            st.warning("Internal error: Please ensure a valid visit is selected.")


# --- Run the main function ---
if __name__ == "__main__":
    main()
