import streamlit as st
import pandas as pd
import whisper # Requires: pip install -U openai-whisper
import os
import tempfile
import logging # For better error tracking if deployed
import re
import json # To format structured data nicely for the prompt
from openai import OpenAI, OpenAIError # Import OpenAI library

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths for mock data
BASE_DIR = os.path.dirname(__file__)
PATIENTS_FILE = os.path.join(BASE_DIR, "patients.csv")
PROVIDERS_FILE = os.path.join(BASE_DIR, "providers.csv")
VISITS_FILE = os.path.join(BASE_DIR, "visits.csv")
DIAGNOSES_FILE = os.path.join(BASE_DIR, "diagnoses.csv")
PROCEDURES_FILE = os.path.join(BASE_DIR, "procedures.csv")

# Choose Whisper model size
WHISPER_MODEL_SIZE = "base"
# Define OpenAI Model
OPENAI_MODEL = "gpt-4o" # Or "gpt-3.5-turbo", etc.

# Define discharge summary sections ORDER MATTERS for the prompt structure
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
# Identify sections expected to be primarily from narrative
NARRATIVE_SECTIONS = [
    "History of Present Illness",
    "Hospital Course",
    "Significant Findings",
    "Discharge Medications",
    "Condition at Discharge",
    "Discharge Instructions/Recommendations",
    "Follow-up",
]


# --- Data Loading Functions ---
@st.cache_data
def load_all_data():
    """Loads all mock data from CSV files, stripping key ID whitespace."""
    # ... (load_all_data function remains the same) ...
    data = {}
    files_to_load = {
        "patients": PATIENTS_FILE,
        "providers": PROVIDERS_FILE,
        "visits": VISITS_FILE,
        "diagnoses": DIAGNOSES_FILE,
        "procedures": PROCEDURES_FILE
    }
    all_files_found = True
    for key, file_path in files_to_load.items():
        try:
            df = pd.read_csv(file_path, dtype=str)
            id_columns = ['PatientID', 'VisitID', 'ProviderID', 'DischargingPhysicianID', 'DiagnosingProviderID', 'PerformingPhysicianID', 'DiagnosisID', 'ProcedureID']
            for col in id_columns:
                 if col in df.columns:
                    df[col] = df[col].str.strip()
            data[key] = df
            for col in data[key].columns:
                if 'date' in col.lower():
                    try:
                        data[key][col] = pd.to_datetime(data[key][col], errors='coerce')
                    except Exception as e:
                        logging.warning(f"Could not parse date column {col} in {key}: {e}")
        except FileNotFoundError:
            st.error(f"Error: Mock data file not found: {file_path}.")
            logging.error(f"Mock data file not found: {file_path}")
            all_files_found = False
            data[key] = pd.DataFrame()
        except Exception as e:
            st.error(f"An error occurred while loading {file_path}: {e}")
            logging.error(f"Error loading {file_path}: {e}")
            all_files_found = False
            data[key] = pd.DataFrame()
    return data if all_files_found else None


# --- Transcription Function ---
@st.cache_resource
def load_whisper_model(model_size=WHISPER_MODEL_SIZE):
    """Loads the specified Whisper model."""
    # ... (load_whisper_model function remains the same) ...
    try:
        logging.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        logging.info(f"Whisper model '{model_size}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model '{model_size}': {e}. Please ensure FFmpeg is installed and the model name is correct.")
        logging.error(f"Error loading Whisper model '{model_size}': {e}")
        st.warning("Transcription functionality will be unavailable.")
        return None

def transcribe_audio(model, audio_path):
    """Transcribes audio file using the loaded Whisper model."""
    # ... (transcribe_audio function remains the same) ...
    if model is None:
        logging.error("Transcription attempted without a loaded model.")
        return "Error: Transcription model not loaded."
    try:
        logging.info(f"Starting transcription for: {audio_path}")
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found at {audio_path}")
            return f"Error: Audio file not found at {audio_path}"
        result = model.transcribe(audio_path, fp16=False)
        logging.info(f"Transcription successful for: {audio_path}")
        return result["text"]
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        logging.error(f"Transcription failed for {audio_path}: {e}")
        return f"Transcription failed. Error: {e}"

# --- Helper Functions ---
def format_date(date_obj):
    """Formats datetime objects, handling NaT."""
    # ... (format_date function remains the same) ...
    if pd.isna(date_obj):
        return "N/A"
    try:
        if isinstance(date_obj, pd.Timestamp):
             return date_obj.strftime('%Y-%m-%d')
        else:
             dt_obj = pd.to_datetime(date_obj, errors='coerce')
             if pd.isna(dt_obj):
                 return "Invalid Date"
             return dt_obj.strftime('%Y-%m-%d')
    except Exception as e:
        logging.warning(f"Date formatting failed for {date_obj}: {e}")
        return str(date_obj) # Fallback

def get_provider_name(provider_id, providers_df):
    """Looks up provider name from ProviderID."""
    # ... (get_provider_name function remains the same) ...
    if provider_id is None or pd.isna(provider_id) or providers_df.empty:
        return "N/A"
    provider = providers_df[providers_df['ProviderID'] == str(provider_id).strip()]
    if not provider.empty:
        return provider['ProviderName'].iloc[0]
    logging.warning(f"Provider ID '{provider_id}' not found in providers table.")
    return f"Unknown ({provider_id})"

# --- LLM Integration Logic ---
def get_openai_api_key():
    """Gets the OpenAI API key from Streamlit secrets or environment variables."""
    # ... (get_openai_api_key function remains the same) ...
    try:
        if hasattr(st, 'secrets') and "openai" in st.secrets and "api_key" in st.secrets["openai"]:
             logging.info("Using OpenAI API key from Streamlit secrets.")
             return st.secrets["openai"]["api_key"]
    except Exception as e:
        logging.warning(f"Could not access Streamlit secrets (normal for local dev): {e}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logging.info("Using OpenAI API key from environment variable.")
        return api_key
    else:
        logging.warning("OpenAI API key not found in secrets or environment variable.")
        return None

def generate_summary_with_llm(structured_data, transcription):
    """Generates discharge summary using OpenAI API."""
    api_key = get_openai_api_key()
    if not api_key:
        st.error("OpenAI API Key not found. Please set it via environment variable (OPENAI_API_KEY) or Streamlit secrets.")
        return "Error: API Key not configured."

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        logging.error(f"OpenAI client initialization failed: {e}")
        return "Error: Failed to initialize OpenAI client."

    # --- Construct the Prompt ---
    structured_data_str = ""
    for key, value in structured_data.items():
        structured_data_str += f"**{key}:**\n"
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                structured_data_str += f"- {sub_key}: {sub_value}\n"
        elif isinstance(value, list) and value:
             structured_data_str += "\n".join([f"- {item}" for item in value]) + "\n"
        elif not isinstance(value, list) or not value: # Handle empty lists or other types
             structured_data_str += f"- {value if value else 'None Recorded'}\n"
        structured_data_str += "\n"

    section_list_str = "\n".join([f"- {section}" for section in SUMMARY_SECTIONS_ORDERED])

    # --- REFINED SYSTEM PROMPT ---
    system_prompt = f"""
You are an expert AI medical scribe creating a hospital discharge summary.
Your goal is to synthesize the provided Structured Data and the Nurse's Voice Note Transcription into a professional, accurate, and well-formatted discharge summary using Markdown.

**Output Format Requirements:**
- Use Markdown Level 3 Headings (`###`) for each section title EXACTLY as listed below.
- The section order MUST follow this specific sequence:
{section_list_str}
- Populate each section with relevant information synthesized from BOTH the Structured Data and the Nurse's Transcription. Prioritize structured data for factual elements (dates, demographics, recorded diagnoses/procedures, provider names). Use the transcription for narrative details (context, findings, medications, instructions, follow-up, condition).
- **Discharge Medications Section:** Format this section ONLY as a Markdown table with columns: `| Medication | Dosage | Frequency | Notes |`. Extract details carefully from the transcription. If details like dosage or frequency are missing in the transcription, leave the corresponding table cell blank or write 'Not specified'. If the transcription refers elsewhere (e.g., "see script"), state that clearly in the 'Notes' column or as a general note.
- **Handling Missing Information:**
    - If NO relevant information can be found for a section (especially narrative ones like History of Present Illness, Significant Findings, Discharge Medications, Follow-up, etc.) in EITHER the structured data OR the transcription, **INCLUDE the section header** (e.g., `### Follow-up`) but add the text `[No information provided in transcription or structured data for this section.]` below it.
    - **DO NOT OMIT SECTION HEADERS** just because information is missing. Explicitly state that information is missing under the relevant header.
- **Handle Conflicts/Vagueness:** If the transcription mentions information conflicting with structured data, include the transcription's version but add a note like `[Nurse Note: ...]` or `[Verification Needed: ...]`. If the transcription is vague (e.g., "follow up soon"), include the vague statement but add a note like `[Nurse Note: Specific timeframe/details needed]`.
- **Professional Tone:** Maintain a concise, professional, and medically appropriate tone.
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
             # Provide a more informative error message to the user
             error_message = "Error: LLM generated an invalid or empty response. This might happen if the audio transcription was very short or unclear. Please review the transcription and try again, or manually edit the summary."
             st.error(error_message)
             return error_message
        return summary.strip()

    except OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        logging.error(f"OpenAI API Error: {e}")
        return f"Error generating summary: {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred calling OpenAI: {e}")
        logging.error(f"Unexpected error calling OpenAI: {e}")
        return f"Error generating summary: {e}"


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Discharge Summary AI")
st.title("👩‍⚕️ Discharge Summary AI Prototype (LLM Powered)")
st.markdown("---")

# --- Load Data ---
mock_data = load_all_data()
if mock_data is None:
    st.error("Failed to load critical mock data. Please check file paths and logs.")
    st.stop()

patients_df = mock_data["patients"]
providers_df = mock_data["providers"]
visits_df = mock_data["visits"]
diagnoses_df = mock_data["diagnoses"]
procedures_df = mock_data["procedures"]

# --- Initialize Session State ---
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'llm_summary' not in st.session_state: # Store LLM output
    st.session_state.llm_summary = ""
if 'current_visit_id' not in st.session_state:
    st.session_state.current_visit_id = None
if 'structured_data_for_llm' not in st.session_state: # Store prepared data
    st.session_state.structured_data_for_llm = {}


# --- Step 1 & 2: Select Patient & Visit ---
st.header("Step 1: Select Patient and Visit")
col1a, col1b = st.columns([1, 2])

with col1a:
    # ... (Patient selection logic remains the same) ...
    try:
        patient_options = {f"{row['Name']} ({row['PatientID']})": row['PatientID'] for index, row in patients_df.iterrows()}
        selected_patient_display = st.selectbox("Select Patient:", options=patient_options.keys(), key="patient_selector")
        selected_patient_id = patient_options[selected_patient_display]
    except Exception as e:
        st.error(f"Error setting up patient selection: {e}")
        selected_patient_id = None
        st.stop()

with col1b:
    # ... (Visit selection and structured data prep logic remains the same) ...
    if selected_patient_id:
        try:
            patient_visits = visits_df[visits_df['PatientID'] == str(selected_patient_id).strip()].sort_values(by='AdmissionDate', ascending=False)
        except Exception as e:
             st.error(f"An error occurred filtering visits: {e}")
             patient_visits = pd.DataFrame()

        if not patient_visits.empty:
            visit_options = {
                f"Visit {row['VisitID']} (Adm: {format_date(row['AdmissionDate'])} - Dis: {format_date(row['DischargeDate'])}) - Dx: {row.get('AdmissionDiagnosis', 'N/A')}": row['VisitID']
                for index, row in patient_visits.iterrows()
            }
            selected_visit_display = st.selectbox("Select Visit:", options=visit_options.keys(), key="visit_selector")
            selected_visit_id = visit_options[selected_visit_display]

            if st.session_state.current_visit_id != selected_visit_id:
                st.session_state.current_visit_id = selected_visit_id
                st.session_state.transcription = ""
                st.session_state.llm_summary = ""
                st.session_state.structured_data_for_llm = {}
                logging.info(f"Changed to Visit ID: {selected_visit_id}. Resetting state.")

                try:
                    patient_info = patients_df[patients_df['PatientID'] == selected_patient_id].iloc[0]
                    visit_info = patient_visits[patient_visits['VisitID'] == selected_visit_id].iloc[0]
                    visit_diagnoses = diagnoses_df[diagnoses_df['VisitID'] == str(selected_visit_id).strip()]
                    visit_procedures = procedures_df[procedures_df['VisitID'] == str(selected_visit_id).strip()]
                    discharging_physician_id = visit_info.get('DischargingPhysicianID', None)
                    discharging_physician_name = get_provider_name(discharging_physician_id, providers_df)

                    st.session_state.structured_data_for_llm = {
                        "Patient Demographics": {
                            "Name": patient_info.get('Name', 'N/A'),
                            "DOB": format_date(patient_info.get('DateOfBirth', '')),
                            "Patient ID": selected_patient_id,
                            "Gender": patient_info.get('Gender', 'N/A')
                        },
                        "Admission Details": {
                            "Visit ID": selected_visit_id,
                            "Admission Date": format_date(visit_info.get('AdmissionDate', '')),
                            "Discharge Date": format_date(visit_info.get('DischargeDate', '')),
                            "Admission Diagnosis": visit_info.get('AdmissionDiagnosis', 'N/A'),
                            "Department": visit_info.get('Department', 'N/A')
                        },
                        "Provider Information": {
                             "Discharging Physician": discharging_physician_name
                        },
                        "Diagnoses Recorded": visit_diagnoses[['DiagnosisDescription', 'DiagnosisCode', 'DiagnosisType']].to_dict('records') if not visit_diagnoses.empty else [],
                        "Procedures Recorded": visit_procedures[['ProcedureName', 'ProcedureDate']].apply(lambda row: f"{row['ProcedureName']} ({format_date(row['ProcedureDate'])})", axis=1).tolist() if not visit_procedures.empty else []
                    }
                    logging.info(f"Prepared structured data for LLM for Visit ID: {selected_visit_id}")

                except Exception as e:
                     st.error(f"An error occurred preparing structured data: {e}")
                     logging.error(f"Error during structured data prep: {e}")
                     st.session_state.structured_data_for_llm = {"error": "Failed to prepare structured data"}

        else:
            st.warning(f"No visit data found for Patient ID: {selected_patient_id}")
            if st.session_state.current_visit_id != f"NO_VISITS_{selected_patient_id}":
                st.session_state.current_visit_id = f"NO_VISITS_{selected_patient_id}"
                st.session_state.transcription = ""
                st.session_state.llm_summary = ""
                st.session_state.structured_data_for_llm = {}

    else:
        st.info("Select a Patient to view Visits.")
        st.session_state.current_visit_id = None
        st.session_state.structured_data_for_llm = {}

st.markdown("---")

# --- Step 3: Upload and Transcribe Audio ---
st.header("Step 2: Upload and Transcribe Audio")
valid_visit_selected = st.session_state.current_visit_id is not None and not str(st.session_state.current_visit_id).startswith("NO_VISITS_")

if valid_visit_selected:
    uploaded_audio = st.file_uploader("Upload Nurse Voice Note (.wav, .mp3, .m4a, .aac, etc.):", type=None, key="audio_uploader")

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button("Transcribe and Generate Summary", key="transcribe_button"):
            st.session_state.llm_summary = "" # Clear previous summary
            st.session_state.transcription = "" # Clear previous transcription

            # 1. Transcription
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio_file:
                tmp_audio_file.write(uploaded_audio.getvalue())
                audio_path = tmp_audio_file.name
            logging.info(f"Temporary audio file saved at: {audio_path}")

            model = load_whisper_model()
            transcription_result = "Error: Transcription model not loaded."
            if model:
                with st.spinner("Transcribing audio..."):
                    transcription_result = transcribe_audio(model, audio_path)

                if "error" not in transcription_result.lower():
                    st.session_state.transcription = transcription_result # Store successful transcription
                    st.success("Transcription complete!")

                    # 2. LLM Summary Generation (only if transcription succeeded)
                    if st.session_state.structured_data_for_llm and "error" not in st.session_state.structured_data_for_llm:
                        with st.spinner("Generating discharge summary draft using AI..."):
                            llm_output = generate_summary_with_llm(
                                st.session_state.structured_data_for_llm,
                                st.session_state.transcription
                            )
                        st.session_state.llm_summary = llm_output # Store LLM output
                        if "error" not in llm_output.lower():
                             st.success("AI summary draft generated!")
                        # Error message is handled within generate_summary_with_llm if needed

                    else:
                        st.error("Cannot generate summary because structured data preparation failed.")
                        st.session_state.llm_summary = "Error: Failed to prepare structured data for LLM."

                else:
                    st.session_state.transcription = transcription_result # Show error message
                    st.error("Transcription failed. Cannot generate summary.")
                    st.session_state.llm_summary = "" # Clear summary on transcription error
            else:
                 st.session_state.transcription = transcription_result # Show model load error
                 st.error("Transcription model failed to load. Cannot generate summary.")
                 st.session_state.llm_summary = ""

            try:
                os.remove(audio_path)
                logging.info(f"Temporary audio file removed: {audio_path}")
            except OSError as e:
                logging.error(f"Error removing temporary file {audio_path}: {e}")

            # Don't rerun here, let the results display

else:
    st.info("Please select a Patient and a valid Visit before uploading audio.")

st.markdown("---")

# --- Step 4: Review Transcription & AI Summary ---
st.header("Step 3: Review Transcription and AI Generated Summary")

col2a, col2b = st.columns(2)

with col2a:
    st.subheader("Audio Transcription")
    st.text_area("Transcription Output:", value=st.session_state.get('transcription', ''), height=600, key="transcription_display", disabled=True)

with col2b:
    st.subheader("AI Generated Discharge Summary Draft")
    # Display LLM summary using markdown
    # Add a container with a specific height and scrollbar for potentially long summaries
    summary_container = st.container(height=600) # Adjust height as needed
    summary_container.markdown(st.session_state.get('llm_summary', '*Summary will appear here after transcription and generation.*'), unsafe_allow_html=True)
    st.caption("Review the AI-generated draft above. You can copy/paste it below to make final edits.")

st.markdown("---")

# --- Step 5: Final Edits (Optional) & Send ---
st.header("Step 4: Final Edits and 'Send'")

st.subheader("Final Editable Summary")
# Use the LLM summary as the default value for editing
final_summary_edit = st.text_area(
    "Make any final corrections or additions here:",
    value=st.session_state.get('llm_summary', ''), # Pre-populate with LLM output
    height=400,
    key="final_summary_edit"
)

if st.button("✅ Mark as Ready for Doctor Review", key="send_button"):
    if valid_visit_selected:
        # In a real application, save `final_summary_edit` content
        st.success(f"Discharge summary for Visit {st.session_state.current_visit_id} marked as ready for physician review! (Simulation)")
        logging.info(f"Summary for Visit {st.session_state.current_visit_id} marked ready.")
        st.balloons()
    else:
        st.warning("Please select a patient and a valid visit before sending.")

