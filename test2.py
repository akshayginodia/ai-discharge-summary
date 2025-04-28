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
OPENAI_MODEL = "gpt-4o"

# Placeholder Hospital Details & Images
APP_HEADER_IMAGE_URL = "https://placehold.co/1400x250/005A9C/FFFFFF/png?text=Medical+AI+Solutions&font=raleway&bold"
HOSPITAL_LOGO_URL = "https://placehold.co/180x60/ffffff/005A9C/png?text=Berlin+Central+Klinik&font=raleway&bold"
HOSPITAL_DETAILS = {
    "Name": "Berlin Central Klinik",
    "Address": "Hauptstrasse 101, 10115 Berlin, Germany",
    "Phone": "+49 30 12345678",
    "Website": "www.berlincentralklinik.example.de"
}

# Target discharge summary sections
SUMMARY_SECTIONS_ORDERED = [
    "Patient Demographics", "Admission Details", "Discharge Diagnosis", "Procedures",
    "History of Present Illness", "Hospital Course", "Significant Findings",
    "Discharge Medications", "Condition at Discharge", "Discharge Instructions/Recommendations",
    "Follow-up", "Provider Information"
]

# --- Core Functions ---

# @st.cache_data
def load_data(file_path):
    """Loads a single CSV file into a DataFrame, handling errors and stripping IDs."""
    try:
        df = pd.read_csv(file_path, dtype=str)
        id_columns = ['PatientID', 'VisitID', 'ProviderID', 'DischargingPhysicianID', 'DiagnosingProviderID', 'PerformingPhysicianID', 'DiagnosisID', 'ProcedureID']
        for col in id_columns:
             if col in df.columns: df[col] = df[col].str.strip()
        for col in df.columns:
            if 'date' in col.lower(): df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError: st.error(f"Error: Mock data file not found: {file_path}."); return None
    except Exception as e: st.error(f"An error occurred while loading {file_path}: {e}"); return None

# @st.cache_resource
def load_whisper_model(model_size=WHISPER_MODEL_SIZE):
    """Loads the specified Whisper model."""
    try:
        logging.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        logging.info(f"Whisper model '{model_size}' loaded successfully.")
        return model
    except Exception as e: st.error(f"Error loading Whisper model '{model_size}': {e}. Ensure FFmpeg is installed."); return None

def transcribe_audio(model, audio_path):
    """Transcribes audio file using the loaded Whisper model."""
    if model is None: return "Error: Transcription model not loaded."
    try:
        logging.info(f"Starting transcription for: {audio_path}")
        if not os.path.exists(audio_path): return f"Error: Audio file not found at {audio_path}"
        result = model.transcribe(audio_path, fp16=False)
        logging.info(f"Transcription successful for: {audio_path}")
        return result["text"]
    except Exception as e: st.error(f"An error occurred during transcription: {e}"); return f"Transcription failed. Error: {e}"

def extract_patient_identifiers(text):
    """Basic extraction of patient name/ID using regex."""
    extracted = {"name": None, "id": None}
    if not text: return extracted
    try:
        name_patterns = [ r'(?:patient name is|patient is|for patient|summary for|note for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:patient id|id)']
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
    mismatch = False; message = ""
    if name_in_audio or id_in_audio:
        logging.info(f"Validating Audio (Name: {name_in_audio}, ID: {id_in_audio}) vs Selection (Name: {selected_patient_name}, ID: {selected_patient_id})")
        if name_in_audio and selected_patient_name and name_in_audio.lower().strip() not in selected_patient_name.lower().strip():
            mismatch = True; message += f"Patient name mismatch ('{name_in_audio}' vs '{selected_patient_name}'). "
        if id_in_audio and selected_patient_id:
            audio_id_norm = id_in_audio.lower().strip(); selected_id_norm = selected_patient_id.lower().strip(); selected_id_num_only = selected_id_norm.replace("pat","")
            if audio_id_norm != selected_id_norm and audio_id_norm != selected_id_num_only:
                mismatch = True; message += f"Patient ID mismatch ('{id_in_audio}' vs '{selected_patient_id}')."
    if mismatch: return False, f"Validation Error: {message.strip()} Check selection or audio."
    else: return True, "Validation passed (or no conflicting identifiers found/matched in audio)."


def get_openai_api_key():
    """Gets the OpenAI API key from Streamlit secrets or environment variables."""
    try:
        if hasattr(st, 'secrets') and "openai" in st.secrets and "api_key" in st.secrets["openai"]: return st.secrets["openai"]["api_key"]
    except Exception as e: logging.warning(f"Could not access Streamlit secrets: {e}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key: return api_key
    else: logging.warning("OpenAI API key not found."); return None

def generate_summary_with_llm(structured_data, transcription, hospital_details):
    """Generates discharge summary using OpenAI API, including hospital header."""
    api_key = get_openai_api_key()
    if not api_key: error_msg = "Error: API Key not configured."; st.error("OpenAI API Key not found."); return error_msg
    try: client = OpenAI(api_key=api_key)
    except Exception as e: error_msg = f"Error: Failed to initialize OpenAI client - {e}"; st.error(error_msg); logging.error(f"OpenAI client init failed: {e}"); return error_msg
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
        response = client.chat.completions.create( model=OPENAI_MODEL, messages=[ {"role": "system", "content": system_prompt}, {"r