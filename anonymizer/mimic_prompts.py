"""
MIMIC-IV specific prompt configuration.

This module contains prompts tailored for MIMIC-IV medical database structure,
with specific references to MIMIC column names and identifiers.

Use with: python anonymize_agentic.py --prompt-config mimic <input>
"""

from dataclasses import dataclass, field
from anonymizer.prompt_config import PromptConfig


@dataclass
class MimicPromptConfig(PromptConfig):
    """
    MIMIC-IV specific prompt configuration.

    These prompts include explicit references to MIMIC-IV specific columns
    like subject_id, hadm_id, stay_id, etc.
    """

    # Phase 1b: Column-level PII detection (CSV only) - MIMIC-specific
    column_detection_prompt: str = field(default="""You are a PII detection agent. Analyze this CSV structure and identify columns that contain Personal Identifiable Information (PII) that should be ENTIRELY redacted.

CSV Column Headers: {columns}

Sample Data (first rows):
{sample_data}

Your task: Identify columns where ALL values should be redacted because the column contains ONLY identifiers or PII.

Common PII columns to look for:
- subject_id, patient_id: Patient identifiers
- hadm_id: Hospital admission ID
- stay_id: ICU/hospital stay identifier
- note_id: Clinical note identifier
- caregiver_id, provider_id: Healthcare provider identifiers
- order_id: Order identifiers
- microevent_id, labevent_id, pharmacy_id, poe_id, emar_id, specimen_id, transfer_id: Various MIMIC event identifiers
- Any column with "id" suffix that links to individuals

DO NOT mark as PII columns:
- Columns with mixed content that are NOT PURE IDENTIFIERS
- Date/time columns
- Medical codes (icd_code, CPT, etc.)
- Generic categorical columns (admission_type, discharge_location, etc.)
- Numeric measurements (lab values, vitals, etc.)
- Sequence numbers (seq_num, poe_seq, note_seq etc.)
- med_rn, gsn_rn, gsn, ndc (these are medical codes, not PII)
- itemid (its a code for eventitem, not PII)
- discharge_location, admission_type, insurance (these are categorical, not PII)
- comments, descriptions, notes
                                         
For EACH column that should be entirely redacted, call:
  redact_column(column_name="exact_column_name", reason="brief reason")

If no columns need full redaction, just respond with "No columns identified for full redaction."
""")

    # Phase 2: PII anonymization for text files (.txt, .hea) - MIMIC-specific
    text_anonymization_prompt: str = field(default="""You are a PII anonymization agent for MIMIC-IV medical data. Analyze this text and redact ALL Personal Identifiable Information (PII).

CRITICAL: You MUST scan the ENTIRE document, even if it's very long. Do NOT skip any sections!
If the document has 100+ lines, you MUST check ALL of them for PIIs.

IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

=== TEXT TO ANALYZE ===
{text_data}
=== END OF TEXT ===

You have the redact_text tool available. For EACH piece of PII you find, call:
  redact_text(text_to_redact="exact PII text") e.g. redact_text(text_to_redact="John Doe")

PII categories to redact (MIMIC-IV specific):
- name: Patient names, physician names, doctor names, staff names, caregiver names
- id: subject_id, hadm_id, stay_id, note_id, caregiver_id, and any other MIMIC identifiers
- address: Physical addresses, street names
- location: Hospital names (Beth Israel Deaconess Medical Center, etc.), specific facility names
- phone: Phone numbers, fax numbers
- email: Email addresses

MIMIC-IV ECG Header (.hea) files may contain:
- Subject IDs in filename references (e.g., "45790175.dat")
- Record identifiers in the first line
- Any numeric IDs that could link to patients
                                           
Example reductions for .hea files:
45790175 12 500 5000 04:57:00 16/07/2131 -> ******** 12 500 5000 04:57:00 16/07/2131
45790175.dat 16 200.0(0)/mV 16 0 19 3475 0 I -> ********.dat 16 200.0(0)/mV 16 0 19 3475 0 I
# <subject_id>: 10045929 -> # <subject_id>: *********

DO NOT redact:
- Dates and times (already shifted)
- Medical terminology, diagnoses, procedures, medications
- Generic locations like "EMERGENCY ROOM", "HOME", "ICU", "WARD"
- Lab values, measurements, vital signs
- ECG technical metadata (sampling rate, gain, ADC resolution, lead names)
- Sequence numbers, lab codes, medical codes (itemid, icd_code, etc.)

IMPORTANT:
- Call redact_text for EACH piece of PII you find
- Use the EXACT text as it appears (preserve spacing and punctuation)
- Redact complete names, not just first or last names
- Be thorough - check for all PII types
""")

    # Phase 2: PII anonymization for CSV - MIMIC-specific
    csv_anonymization_prompt: str = field(default="""You are a PII anonymization agent. Analyze this CSV data and redact ALL Personal Identifiable Information (PII).{skipped_columns}

IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

CSV Data:
{csv_data}

You have TWO tools available:

1. redact_text - For individual PII values:
   redact_text(text_to_redact="exact PII text", row_index=N, column_name="column")

2. redact_column - For entire columns containing PII (more efficient):
   redact_column(column_name="column_name", reason="why this column is PII")
   Use this when you notice a column contains identifiers in ALL rows (e.g., subject_id, hadm_id, note_id)

PII categories to redact (BE THOROUGH - check EVERY occurrence):
- name: Patient names, physician names, doctor names, family member names, caregiver names, any other personal names
- address: Physical addresses, street addresses, facility names, location names (e.g. hospital names)
- id: Patient IDs, medical record numbers, unit numbers, order ids, note ids, subject ids and any other numeric identifiers
- phone: Phone numbers
- fax: Fax numbers
- email: Email addresses
- ids: Any other identifiers that can link to an individual or organization (provider IDs, account numbers, caregiver IDs, etc.)
- other: Any other specific information that can identify an individual

EFFICIENCY TIP: If you notice a column like "subject_id" or "hadm_id" contains identifiers in every row,
use redact_column ONCE instead of calling redact_text for each row!

DO NOT redact:
- Dates and times (already shifted)
- Medical terminology, diagnoses, procedures, medications
- Generic locations like "EMERGENCY ROOM", "HOME", "ICU"
- Sequence numbers, lab codes, medical codes

IMPORTANT:
- Call redact_text for EACH piece of PII you find
- Use the EXACT text as it appears (preserve spacing)
- Use absolute row indices (0-based)
- Redact the specific PII text, not the entire cell
""")

    # Image anonymization prompt - MIMIC-specific
    image_anonymization_prompt: str = field(default="""Analyze this medical image from MIMIC-IV and identify ALL text that contains Personal Identifiable Information (PII) that should be redacted for patient privacy.

The following texts were detected in the image by OCR:
{ocr_text_list}

For EACH piece of PII you identify:
1. text: Copy the EXACT text as it appears (must match one of the OCR texts above as closely as possible)
2. type: Classify the PII type

PII categories to identify (MIMIC-IV specific):
- name: Patient names, physician/doctor names, any person names
- date_of_birth: Dates of birth, DOB
- id_number: Patient IDs (subject_id), medical record numbers (MRN), accession numbers, study IDs, hadm_id, stay_id
- address: Physical addresses, street addresses
- location: Hospital names (Beth Israel Deaconess Medical Center, etc.), clinic names, facility names, cities, institutions
- phone: Phone numbers, fax numbers
- email: Email addresses
- dates: Specific dates (admission date, study date, exam date, charttime, etc.) that could identify a patient

IMPORTANT:
- Focus on text that could identify a specific patient or person
- Include ALL identifying information, not just obvious names
- MIMIC identifiers like subject_id, hadm_id, stay_id are PII
- Hospital/facility names are PII (location)
- Dates associated with studies or admissions are PII
- Generic medical terms, measurements, and MIMIC codes are NOT PII
""")

    # Image verification prompt - MIMIC-specific
    image_verification_prompt: str = field(default="""Carefully analyze this medical image from MIMIC-IV that has been redacted (black rectangles cover sensitive information).

Your task is to verify that ALL Personal Identifiable Information (PII) has been properly redacted.

Look for ANY remaining PII that is still visible and NOT covered by black redaction boxes:
- Patient names, physician names, any person names
- Dates of birth, birth dates
- Patient IDs (subject_id), medical record numbers (MRN), accession numbers, study IDs, hadm_id, stay_id
- Physical addresses, street addresses
- Hospital names (Beth Israel Deaconess Medical Center, etc.), clinic names, facility names, institution names
- Phone numbers, fax numbers
- Email addresses
- Specific dates (admission dates, study dates, exam dates, charttime) that could identify a patient

IMPORTANT:
- Black rectangles are intentional redactions - ignore them
- Focus on ANY text that is STILL VISIBLE and contains PII
- Even partially visible PII should be reported
- If text is visible but clearly NOT PII (medical terms, measurements, MIMIC codes), do not report it

Set is_clean to TRUE only if you are confident that NO PII remains visible.
Set is_clean to FALSE if you find ANY remaining PII that needs additional redaction.

Be thorough - missing even one PII element is a privacy violation.
""")

    # Phase 3: Text verification prompt - MIMIC-specific
    text_verification_prompt: str = field(default="""You are a verification agent for MIMIC-IV medical data anonymization.
Compare the ORIGINAL and ANONYMIZED text below and identify any issues.

CRITICAL: You MUST check the ENTIRE document carefully!
Do NOT stop at the first few lines - scan ALL the way to the end.

=== ORIGINAL TEXT ===
{original_text}

=== ANONYMIZED TEXT ===
{anonymized_text}

Time offset used: {time_offset} days

You have TWO tools available:
1. shift_datetime - to fix unshifted dates
2. redact_text - to redact PII that was missed

Your tasks (BE THOROUGH - check EVERY line):

1. CHECK FOR UNSHIFTED DATES: Look for dates in ANONYMIZED that are identical to ORIGINAL.
   All dates should be shifted by {time_offset} days.
   → Use shift_datetime(datetime_str, offset_days={time_offset}) to fix

2. CHECK FOR UNREDACTED PII (MIMIC-IV specific): Look for PII that appears UNCHANGED:
   - Patient names, doctor names, staff names, caregiver names
   - MIMIC identifiers: subject_id, hadm_id, stay_id, note_id values
   - Phone numbers, fax numbers, email addresses
   - Hospital names (Beth Israel Deaconess Medical Center, etc.)
   - Physical addresses
   - ECG record IDs in .hea files (numeric IDs like "45790175")
   → Use redact_text(text_to_redact="the PII text") to fix

IMPORTANT:
- Compare ORIGINAL vs ANONYMIZED to identify issues
- Only redact actual PII, not medical content
- Check the ENTIRE document, not just the first page

Call the appropriate tool for each issue you find. When done, summarize your findings.
""")

    # PDF anonymization prompt - MIMIC-specific
    pdf_anonymization_prompt: str = field(default="""Analyze the following texts extracted from a MIMIC-IV medical document and identify which ones contain Personal Identifiable Information (PII).

Texts found in the document:
{text_list}

PII categories to identify (MIMIC-IV specific):
- name: Patient names, physician/doctor names
- date_of_birth: Dates of birth
- id_number: Patient IDs (subject_id), medical record numbers, hadm_id, stay_id, note_id, all other potentially identification numbers
- address: Physical addresses
- location: Locations, e.g. cities, hospital names (Beth Israel Deaconess Medical Center, etc.)
- phone: Phone numbers
- email: Email addresses
- dates: Other specific dates (admission, discharge, study dates, charttime, etc.)

For each text that contains PII, provide:
1. text: The EXACT text as it appears above (must match exactly)
2. type: The PII category

Only include texts that actually contain PII. If a text is just a medical term, MIMIC code, or general information, do not include it.
""")


# Create default MIMIC prompt config instance
MIMIC_PROMPT_CONFIG = MimicPromptConfig()
