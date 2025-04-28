import streamlit as st
import pandas as pd
import whisper
import os
import tempfile
import logging
import re
import imageio_ffmpeg as ffmpeg
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
OPENAI_MODEL = "gpt-4o"

# --- UPDATED: Placeholder Hospital Details & Images ---
# Using a more generic German-sounding name (in English) and medical theme
APP_HEADER_IMAGE_URL = "https://placehold.co/1400x250/005A9C/FFFFFF/png?text=Medical+AI+Solutions&font=raleway&bold"
HOSPITAL_LOGO_URL = "https://placehold.co/180x60/ffffff/005A9C/png?text=Berlin+Central+Klinik&font=raleway&bold" # White background logo
HOSPITAL_DETAILS = {
    "Name": "Berlin Central Klinik", # Fictional but plausible English name
    "Address": "Hauptstrasse 101, 10115 Berlin, Germany", # Example address
    "Phone": "+49 30 12345678", # Example phone
    "Website": "www.berlincentralklinik.example.de" # Example website
}
# --- END UPDATED ---


# Target discharge summary sections based on user document
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

# Point the ffmpeg wrapper at the vendored binary
os.environ["FFMPEG_BINARY"] = ffmpeg.get_ffmpeg_exe()

# --- Core Functions ---

# @st.cache_data
def load_data(file_path):
    """Loads a single CSV file into a DataFrame, handling errors and stripping IDs."""
    try:
        df = pd.read_csv(file_path, dtype=str)
        id_columns = ['PatientID', 'VisitID', 'ProviderID', 'DischargingPhysicianID',
                      'DiagnosingProviderID', 'PerformingPhysicianID', 'DiagnosisID', 'ProcedureID']
        for col in id_columns:
             if col in df.columns: df[col] = df[col].str.strip()
        for col in df.columns:
            if 'date' in col.lower(): df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: Mock data file not found: {file_path}.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading {file_path}: {e}")
        return None

# @st.cache_resource
def load_whisper_model(model_size=WHISPER_MODEL_SIZE):
    """Loads the specified Whisper model."""
    try:
        logging.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        logging.info(f"Whisper model '{model_size}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model '{model_size}': {e}. Ensure FFmpeg is installed.")
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
        return f"Transcription failed. Error: {e}"

def extract_patient_identifiers(text):
    """Basic extraction of patient name/ID using regex."""
    extracted = {"name": None, "id": None}
    if not text: return extracted
    try:
        name_patterns = [
            r'(?:patient name is|patient is|for patient|summary for|note for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:patient id|id)'
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match: extracted["name"] = name_match.group(1).strip(); break
        id_patterns = [ r'(?:patient id|id)\s+([A-Z]{3}\d+)', r'(?:patient id|id)\s+(\d+)']
        for pattern in id_patterns:
            id_match = re.search(pattern, text, re.IGNORECASE)
            if id_match: extracted["id"] = id_match.group(1).strip(); break
    except Exception as e: logging.error(f"Error during identifier extraction: {e}")
    logging.info(f"Extracted identifiers from transcription: {extracted}")
    return extracted

def validate_transcription_vs_selection(transcription, selected_patient_id, selected_patient_name):
    """Compares extracted identifiers with selected patient data."""
    if not transcription or not selected_patient_id or not selected_patient_name:
        return True, "Validation skipped: Missing transcription or selection data."
    extracted = extract_patient_identifiers(transcription)
    name_in_audio = extracted.get("name")
    id_in_audio = extracted.get("id")
    mismatch = False
    message = ""
    if name_in_audio or id_in_audio:
        logging.info(f"Validating Audio (Name: {name_in_audio}, ID: {id_in_audio}) vs Selection (Name: {selected_patient_name}, ID: {selected_patient_id})")
        if name_in_audio and selected_patient_name and name_in_audio.lower().strip() not in selected_patient_name.lower().strip():
            mismatch = True
            message += f"Patient name mismatch ('{name_in_audio}' vs '{selected_patient_name}'). "
        if id_in_audio and selected_patient_id:
            audio_id_norm = id_in_audio.lower().strip()
            selected_id_norm = selected_patient_id.lower().strip()
            selected_id_num_only = selected_id_norm.replace("pat","")
            if audio_id_norm != selected_id_norm and audio_id_norm != selected_id_num_only:
                mismatch = True
                message += f"Patient ID mismatch ('{id_in_audio}' vs '{selected_patient_id}')."
    if mismatch: return False, f"Validation Error: {message.strip()} Check selection or audio."
    else: return True, "Validation passed (or no conflicting identifiers found/matched in audio)."


def get_openai_api_key():
    """Gets the OpenAI API key from Streamlit secrets or environment variables."""
    try:
        if hasattr(st, 'secrets') and "openai" in st.secrets and "api_key" in st.secrets["openai"]:
             return st.secrets["openai"]["api_key"]
    except Exception as e: logging.warning(f"Could not access Streamlit secrets: {e}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key: return api_key
    else: logging.warning("OpenAI API key not found."); return None

def generate_summary_with_llm(structured_data, transcription, hospital_details):
    """Generates discharge summary using OpenAI API, including hospital header."""
    api_key = get_openai_api_key()
    if not api_key:
        error_msg = "Error: API Key not configured."
        st.error("OpenAI API Key not found. Set OPENAI_API_KEY env var or Streamlit secrets.")
        return error_msg

    try: client = OpenAI(api_key=api_key)
    except Exception as e:
        error_msg = f"Error: Failed to initialize OpenAI client - {e}"
        st.error(error_msg); logging.error(f"OpenAI client init failed: {e}"); return error_msg

    # --- Construct the Prompt ---
    structured_data_str = ""
    for key, value in structured_data.items():
        structured_data_str += f"**{key}:**\n"
        if isinstance(value, dict):
            for sub_key, sub_value in value.items(): structured_data_str += f"- {sub_key}: {sub_value}\n"
        elif isinstance(value, list) and value: structured_data_str += "\n".join([f"- {item}" for item in value]) + "\n"
        elif value: structured_data_str += f"- {value}\n"
        else: structured_data_str += f"- None Recorded\n"
        structured_data_str += "\n"

    section_list_str = "\n".join([f"- {section}" for section in SUMMARY_SECTIONS_ORDERED])

    # --- REFINED SYSTEM PROMPT with Hospital Header Instruction ---
    system_prompt = f"""
You are an expert AI medical scribe creating a hospital discharge summary.
Your goal is to synthesize the provided Structured Data and the Nurse's Voice Note Transcription into a professional, accurate, and well-formatted discharge summary using Markdown.

**Output Format Requirements:**
- **Hospital Header:** Start the document with a header containing the hospital details provided (Name, Address, Phone, Logo URL) using Markdown. Format it clearly, perhaps with the logo first, then name, address, phone. Example: `![Hospital Logo]({HOSPITAL_LOGO_URL}?style=rounded)\n## {hospital_details['Name']}\n{hospital_details['Address']} | {hospital_details['Phone']}\n***\n` (Ensure the logo URL is included correctly in the Markdown image tag).
- **Section Headers:** Use Markdown Level 3 Headings (`###`) for each subsequent section title EXACTLY as listed below.
- **Section Order:** The section order MUST follow this specific sequence AFTER the hospital header:
{section_list_str}
- **Content Synthesis:** Populate each section with relevant information synthesized from BOTH the Structured Data and the Nurse's Transcription. Prioritize structured data for factual elements (dates, demographics, recorded diagnoses/procedures, provider names). Use the transcription for narrative details (context, findings, medications, instructions, follow-up, condition).
- **Discharge Medications Section:** Format this section ONLY as a Markdown table with columns: `| Medication | Dosage | Frequency | Notes |`. Extract details carefully from the transcription. If details like dosage or frequency are missing, leave the cell blank or write 'Not specified'. If the transcription refers elsewhere (e.g., "see script"), state that clearly in the 'Notes' column.
- **Handling Missing Information:** If NO relevant information can be found for a section in EITHER the structured data OR the transcription, INCLUDE the section header (e.g., `### Follow-up`) but add the text `[No information provided in transcription or structured data for this section.]` below it. Do not omit headers.
- **Handling Conflicts/Vagueness:** If transcription conflicts with structured data, include the transcription's version but add `[Nurse Note: Verification Needed - structured data shows X]`. If transcription is vague, include it but add `[Nurse Note: Specific timeframe/details needed]`.
- **Professional Tone:** Maintain a concise, professional, medically appropriate tone.
- **Start Directly:** Begin the output directly with the Hospital Header as instructed. Do not add other introductory phrases.
"""

    user_prompt = f"""
Generate the discharge summary based on the following information, adhering strictly to the formatting and content instructions in the system prompt:

**Hospital Details for Header:**
```json
{json.dumps(hospital_details)}
```

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
            max_tokens=2800
        )
        summary = response.choices[0].message.content
        logging.info("Received response from OpenAI.")
        # Check if response includes expected structure (header + first section)
        # Use a simpler check for the header part
        if not summary or f"## {hospital_details['Name']}" not in summary or f"### {SUMMARY_SECTIONS_ORDERED[0]}" not in summary:
             logging.warning(f"LLM response might be invalid or missing structure: {summary[:300]}...")
             error_message = "Error: LLM generated an invalid or improperly formatted response. Please review transcription and try again, or edit manually."
             st.warning(error_message)
             return error_message
        return summary.strip()

    except OpenAIError as e:
        error_msg = f"Error: OpenAI API Error - {e}"
        st.error(error_msg); logging.error(f"OpenAI API Error: {e}"); return error_msg
    except Exception as e:
        error_msg = f"Error: An unexpected error occurred calling OpenAI - {e}"
        st.error(error_msg); logging.error(f"Unexpected error calling OpenAI: {e}"); return error_msg

# --- Helper Functions ---
def format_date_display(date_obj):
    """Formats datetime objects for display, handling NaT."""
    if pd.isna(date_obj): return "N/A"
    try: return pd.to_datetime(date_obj).strftime('%Y-%m-%d')
    except: return "Invalid Date"

def get_display_provider_name(provider_id, providers_df):
    """Looks up provider name from ProviderID for display."""
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
            background-image: linear-gradient(to bottom right, #e9f2f7, #ffffff); /* Lighter gradient */
        }
        /* Add padding and subtle border to the main content block */
        .main .block-container {
            padding: 1.5rem 2.5rem 3rem 2.5rem; /* Adjusted padding */
            background-color: rgba(255, 255, 255, 0.95); /* Slightly less transparent */
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Softened shadow */
            border: 1px solid #dfe6e9; /* Softer border */
            max-width: 1600px;
            margin: auto;
        }
        /* Style headers */
        h1 {
            color: #005A9C;
            border-bottom: 2px solid #005A9C;
            padding-bottom: 0.3em;
            margin-top: 0.5em; /* Reduce space above title */
            margin-bottom: 0.8em; /* Adjust space below title */
            text-align: center;
        }
        h2 {
            color: #0073B7;
            margin-top: 2em;
            margin-bottom: 1em;
            border-bottom: 1px solid #b2bec3; /* Softer border color */
            padding-bottom: 0.3em;
            font-size: 1.6rem; /* Slightly smaller section headers */
        }
         h3 {
            color: #2d3436; /* Darker grey for subheaders */
            margin-top: 1.2em;
            margin-bottom: 0.6em;
            font-size: 1.15rem; /* Slightly smaller subheaders */
            font-weight: bold;
        }
        /* Style buttons */
        .stButton>button {
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            border: 1px solid #005A9C;
            color: white;
            background-color: #0073B7;
            transition: background-color 0.3s ease, color 0.3s ease;
            margin-top: 1em;
        }
        .stButton>button:hover {
            background-color: #005A9C;
            color: white;
        }
        .stButton>button:disabled {
            background-color: #dfe6e9; /* Lighter disabled background */
            color: #636e72; /* Darker disabled text */
            border-color: #b2bec3;
        }
        /* Style containers used for display */
        .stContainer {
            border: 1px solid #dfe6e9 !important; /* Softer border */
            border-radius: 8px !important;
            padding: 1rem !important;
            background-color: #ffffff;
        }
        /* Style captions */
        .stCaption {
            color: #636e72; /* Slightly darker caption */
            font-style: italic;
            margin-top: 0.2em;
            margin-bottom: 1em;
        }
        /* Ensure markdown tables look decent */
        table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
        th, td { border: 1px solid #dfe6e9; padding: 8px; text-align: left; vertical-align: top; }
        th { background-color: #f0f8ff; font-weight: bold; color: #2d3436; } /* AliceBlue header */
        /* Improve spacing for select boxes */
        .stSelectbox { margin-bottom: 0.5rem; }
        /* Style the expander for transcription */
        .stExpander { border: 1px solid #dfe6e9; border-radius: 8px; margin-top: 1rem; background-color: #f8f9fa; }
        .stExpander header { font-weight: bold; color: #2d3436; font-size: 0.95rem; }
        /* Style the header image */
        div[data-testid="stImage"] > img {
             max-height: 200px; /* Limit header image height */
             object-fit: cover; /* Crop image nicely if needed */
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Header Image ---
    if APP_HEADER_IMAGE_URL:
        # Use use_container_width=True to fix deprecation warning
        st.image(APP_HEADER_IMAGE_URL, use_column_width='always') # Corrected parameter name
    # --- END Header Image ---

    st.title("Discharge Summary AI Assistant")
    st.caption("Generate discharge summary drafts from voice notes and structured data.")

    # --- Load Data ---
    # ... (Data loading remains the same) ...
    all_data = {}
    data_loaded_successfully = True
    for name, path in DATA_FILES.items():
        df = load_data(path)
        if df is None: data_loaded_successfully = False; break
        all_data[name] = df
    if not data_loaded_successfully: st.error("App start failed: data files missing."); st.stop()
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
    if 'edited_summary' not in st.session_state: st.session_state.edited_summary = ""

    # --- Section 1: Select Patient & Visit ---
    st.header("📄 Select Patient and Visit")
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
                 st.session_state.edited_summary = ""
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
                    st.session_state.edited_summary = ""
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
                         st.session_state.structured_data_for_llm = {"error": "Failed to prepare data"} # Updated error message
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
                         st.session_state.structured_data_for_llm = {"error": "Failed to prepare data"} # Updated error message

            else: # No visits found
                st.warning(f"No visit data found for Patient ID: {st.session_state.current_patient_id}")
                if st.session_state.current_visit_id is not None:
                    st.session_state.current_visit_id = None; st.session_state.transcription = ""; st.session_state.llm_summary = ""; st.session_state.edited_summary = ""; st.session_state.structured_data_for_llm = {}; st.session_state.validation_error = None; st.session_state.processing_complete = False
        else: # No patient selected
            st.info("Select a Patient to view Visits.")
            if st.session_state.current_visit_id is not None:
                 st.session_state.current_visit_id = None; st.session_state.transcription = ""; st.session_state.llm_summary = ""; st.session_state.edited_summary = ""; st.session_state.structured_data_for_llm = {}; st.session_state.validation_error = None; st.session_state.processing_complete = False

    # Removed the st.markdown("---") here

    # --- Section 2: Upload Audio & Process ---
    st.header("🎙️ Upload Audio & Generate Summary")
    valid_visit_selected = st.session_state.current_visit_id is not None

    if valid_visit_selected:
        up_col1, up_col2 = st.columns([3,1])
        with up_col1:
            uploaded_audio = st.file_uploader("Upload Nurse Voice Note (.wav, .mp3, .m4a, .aac, etc.):", type=None, key="audio_uploader", label_visibility="collapsed")
        with up_col2:
            process_button_clicked = st.button("✨ Generate Summary", key="process_button", disabled=(uploaded_audio is None), use_container_width=True)

        if st.session_state.validation_error:
            st.error(st.session_state.validation_error)

        if uploaded_audio is not None:
            st.audio(uploaded_audio)

            if process_button_clicked:
                # Reset state for new processing run
                st.session_state.llm_summary = ""
                st.session_state.transcription = ""
                st.session_state.validation_error = None
                st.session_state.processing_complete = False
                st.session_state.edited_summary = ""

                # --- 1. Transcription ---
                transcription_result = "Error: Could not process audio file."
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio_file:
                    tmp_audio_file.write(uploaded_audio.getvalue())
                    audio_path = tmp_audio_file.name
                model = load_whisper_model()
                if model:
                    with st.spinner("Transcribing audio..."):
                        transcription_result = transcribe_audio(model, audio_path)
                else:
                    transcription_result = "Error: Transcription model failed to load."
                st.session_state.transcription = transcription_result
                try: os.remove(audio_path)
                except OSError as e: logging.error(f"Error removing temp file {audio_path}: {e}")

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
                        if st.session_state.structured_data_for_llm and "error" not in st.session_state.structured_data_for_llm:
                            with st.spinner("Generating discharge summary draft using AI..."):
                                llm_output = generate_summary_with_llm(
                                    st.session_state.structured_data_for_llm,
                                    st.session_state.transcription,
                                    HOSPITAL_DETAILS # Pass hospital details
                                )
                            st.session_state.llm_summary = llm_output
                            if "error" not in llm_output.lower():
                                 st.success("AI summary draft generated!")
                                 st.session_state.processing_complete = True
                                 st.session_state.edited_summary = llm_output # Pre-fill edit area
                            else:
                                 st.session_state.edited_summary = "" # Don't pre-fill edit area if LLM error
                        else:
                            st.error("Cannot generate summary: data preparation failed.") # Updated error message
                            st.session_state.llm_summary = "Error: Failed to prepare data for LLM." # Updated error message
                            st.session_state.edited_summary = ""
                    else:
                        # Validation failed
                        st.session_state.validation_error = validation_msg
                        st.session_state.llm_summary = ""
                        st.session_state.edited_summary = ""
                else:
                    # Transcription failed
                    st.error(f"Transcription failed. Cannot generate summary. Details: {transcription_result}")
                    st.session_state.llm_summary = ""
                    st.session_state.edited_summary = ""

                # Rerun needed to display results/errors
                st.rerun()

        # --- Display Transcription Expander ---
        if st.session_state.transcription:
             with st.expander("View Audio Transcription", expanded=False):
                 st.text_area("", value=st.session_state.get('transcription', ''), height=200, key="transcription_display_step2", disabled=True, label_visibility="collapsed")

    else: # No visit selected
        st.info("Please select a Patient and Visit before uploading audio.")

    # Removed the st.markdown("---") here

    # --- Section 3: Review, Edit & Preview Summary ---
    st.header("📝 Review, Edit, and Preview Summary")

    # Use columns for Edit Area and Live Preview
    edit_col, preview_col = st.columns(2)

    # Determine initial value for edit area based on state
    # Use edited_summary state if available, otherwise use llm_summary
    edit_value = st.session_state.edited_summary if st.session_state.edited_summary else st.session_state.llm_summary
    if st.session_state.validation_error or "error" in st.session_state.get('llm_summary', '').lower():
         edit_value = "" # Don't pre-fill if there was an error

    with edit_col:
        st.subheader("Editable Summary")
        # The text area widget itself holds the current edited state
        # Use on_change to update the preview dynamically
        edited_summary_text = st.text_area(
            "", # No label needed due to subheader
            value=edit_value, # Initialize with LLM output or previous edits
            height=700, # Increased height
            key="final_summary_edit_area",
            label_visibility="collapsed",
            # Disable if processing hasn't completed successfully
            disabled=not st.session_state.processing_complete or bool(st.session_state.validation_error) or "error" in st.session_state.get('llm_summary', '').lower(),
            on_change=lambda: st.session_state.update(edited_summary=st.session_state.final_summary_edit_area) # Update state on change
        )
        st.caption("Make final corrections here. The preview on the right will update as you type.")

    with preview_col:
        st.subheader("Formatted Preview")
        # Display status messages or the formatted preview
        preview_content = ""
        # Use the edited_summary state for the preview
        current_edited_text = st.session_state.get('edited_summary', '') # Get current edited text

        if st.session_state.validation_error:
            preview_content = f"**Validation Error:**\n\n{st.session_state.validation_error}\n\n*Summary generation skipped.*"
        elif not st.session_state.processing_complete and "error" in st.session_state.get('transcription','').lower():
             preview_content = f"*Transcription failed: {st.session_state.transcription}*"
        elif not st.session_state.processing_complete and not st.session_state.transcription:
             preview_content = '*Upload audio and click "Transcribe and Generate Summary" first.*'
        elif "error" in st.session_state.get('llm_summary','').lower() and not current_edited_text: # Show LLM error only if edit area is also empty
             preview_content = f"*LLM generation failed:*\n\n{st.session_state.llm_summary}"
        elif not current_edited_text: # If edit area is empty
             preview_content = "*Edit summary on the left to see preview.*"
        else:
             preview_content = current_edited_text # Use content from text_area state

        # Display the live preview using markdown
        preview_container = st.container(height=700, border=True) # Match height
        preview_container.markdown(preview_content, unsafe_allow_html=True)
        st.caption("This preview reflects your edits.")


    st.markdown("---") # Add divider before button

    # --- Finalize Button ---
    # Disable button if processing hasn't completed successfully or validation failed
    finalize_disabled = not st.session_state.processing_complete or bool(st.session_state.validation_error) or "error" in st.session_state.get('llm_summary', '').lower()

    if st.button("✅ Mark as Ready for Doctor Review", key="send_button", disabled=finalize_disabled, use_container_width=True):
        if valid_visit_selected:
            # In a real application, save `edited_summary_text` (from session state) content
            final_content_to_send = st.session_state.get('edited_summary', '')
            st.success(f"Discharge summary for Visit {st.session_state.current_visit_id} marked as ready! (Simulation)")
            logging.info(f"Summary for Visit {st.session_state.current_visit_id} marked ready.")
            # Optionally display the final saved content again
            # st.subheader("Final Content Sent:")
            # st.markdown(final_content_to_send)
            st.balloons()
        else:
            st.warning("Internal error: Please ensure a valid visit is selected.")


# --- Run the main function ---
if __name__ == "__main__":
    main()
