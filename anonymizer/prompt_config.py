"""
Prompt configuration for agentic processors.

This module defines customizable prompt templates that can be modified
via the web interface to adapt the anonymization behavior.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class PromptConfig:
    """
    Configuration for customizable prompts used in agentic processors.

    All prompts support template variables that are filled in at runtime:
    - {columns}: CSV column headers
    - {sample_data}: Sample data preview
    - {csv_data}: Full CSV data for processing
    - {text_data}: Text content for processing
    - {time_offset}: Time offset in days
    - {comparison_data}: Original vs anonymized comparison
    - {skipped_columns}: Columns already redacted
    """

    # Column-level PII detection (CSV/Excel)
    column_detection_prompt: str = field(default="""You are a PII detection agent. Analyze this CSV structure and identify columns that contain Personal Identifiable Information (PII) that should be ENTIRELY redacted.

CSV Column Headers: {columns}

Sample Data (first rows):
{sample_data}

Your task: Identify columns where ALL values should be redacted because the column contains ONLY identifiers or PII.

Common PII columns to look for:
- Columns containing patient identifiers (e.g., patient IDs, medical record numbers)
- Columns containing admission or visit identifiers
- Columns containing provider/caregiver identifiers
- Columns containing document or note identifiers
- Any column with "id" suffix that links to individuals
- Columns that only contain names, SSN, phone numbers, addresses, email

DO NOT mark as PII columns:
- Columns with mixed content that are NOT PURE IDENTIFIERS
- Date/time columns (already handled by time shifting)
- Medical codes (ICD, CPT, etc.)
- Generic categorical columns (admission_type, discharge_location, etc.)
- Numeric measurements (lab values, vitals, etc.)

For EACH column that should be entirely redacted, call:
  redact_column(column_name="exact_column_name", reason="brief reason")

If no columns need full redaction, just respond with "No columns identified for full redaction."
""")

    # PII anonymization for CSV/Excel
    csv_anonymization_prompt: str = field(default="""You are a PII anonymization agent. Analyze this CSV data and redact ALL Personal Identifiable Information (PII).{skipped_columns}

CRITICAL: You MUST scan the ENTIRE content of each cell, even if it's very long. Do NOT skip any PIIs!

IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

CSV Data:
{csv_data}

You have TWO tools available:

1. redact_text - For individual PII values:
   redact_text(text_to_redact="exact PII text", row_index=N, column_name="column")

2. redact_column - For entire columns containing PII (more efficient):
   redact_column(column_name="column_name", reason="why this column is PII")
   Use this when you notice a column contains identifiers in ALL rows (e.g., patient_id, admission_id, note_id)

PII categories to redact (BE THOROUGH - check EVERY occurrence):
- name: Patient names, physician names, doctor names, family member names, caregiver names, any other personal names
- address: Physical addresses, street addresses, facility names, location names (e.g. hospital names)
- id: Patient IDs, medical record numbers, unit numbers, order IDs, note IDs, and any other numeric identifiers
- phone: Phone numbers
- fax: Fax numbers
- email: Email addresses
- ids: Any other identifiers that can link to an individual or organization (provider IDs, account numbers, caregiver IDs, etc.)
- other: Any other specific information that can identify an individual

EFFICIENCY TIP: If you notice a column contains identifiers in every row (e.g., patient_id, admission_id),
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

    # PII anonymization for text/Word files
    text_anonymization_prompt: str = field(default="""You are a PII anonymization agent. Analyze this medical text and redact ALL Personal Identifiable Information (PII).

CRITICAL: You MUST scan the ENTIRE document, even if it's very long. Do NOT skip any sections!
If the document has 100+ lines, you MUST check ALL of them for PIIs.

IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

=== TEXT TO ANALYZE ===
{text_data}
=== END OF TEXT ===

You have the redact_text tool available. For EACH piece of PII you find, call:
  redact_text(text_to_redact="exact PII text")

PII categories to redact:
- Patient names, physician names, doctor names, staff names
- Physical addresses, street names, specific facility/hospital names
- Patient IDs, medical record numbers, unit numbers
- Phone numbers, fax numbers
- Email addresses

DO NOT redact:
- Dates and times (already shifted)
- Medical terminology, diagnoses, procedures, medications
- Generic locations like "EMERGENCY ROOM", "HOME", "ICU", "WARD"
- Lab values, measurements, vital signs
- Sequence numbers, lab codes, medical codes

IMPORTANT:
- Call redact_text for EACH piece of PII you find
- Use the EXACT text as it appears (preserve spacing and punctuation)
- Redact complete names, not just first or last names
- Be thorough - check for all PII types
""")

    # Verification prompt for CSV/Excel
    csv_verification_prompt: str = field(default="""You are a verification agent for medical data anonymization.
Compare the ORIGINAL and ANONYMIZED data below and identify any issues.

CRITICAL: You MUST check the ENTIRE content carefully, especially in long text fields!
Do NOT stop at the first few lines - scan ALL the way to the end of each field.

{comparison_data}

Time offset used: {time_offset} days

You have TWO tools available:
1. shift_datetime - to fix unshifted dates
2. redact_text - to redact PII that was missed

Your tasks (BE THOROUGH - check EVERY line of long text fields):

1. CHECK FOR UNSHIFTED DATES: Look for dates in the ANONYMIZED data that are identical to the ORIGINAL.
   All dates should be shifted by {time_offset} days.
   → Use shift_datetime(datetime_str, offset_days={time_offset}) to fix

2. CHECK FOR UNREDACTED PII: Look for PII that appears UNCHANGED in the anonymized version:
   - Patient names, doctor names, staff names (e.g., "Laura Martinez", "Dr. Smith")
   - Phone numbers (e.g., "555-123-4567")
   - Fax numbers, email addresses
   - Specific addresses or facility names that identify location
   → Use redact_text(text_to_redact, row_index, column_name) to fix

IMPORTANT:
- Compare ORIGINAL vs ANONYMIZED to identify issues
- Only redact actual PII, not medical content
- Restore any medical terms that were incorrectly redacted

Call the appropriate tool for each issue you find. When done, summarize your findings.
""")

    # Verification prompt for text/Word files
    text_verification_prompt: str = field(default="""You are a verification agent for medical data anonymization.
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

Your tasks (BE THOROUGH - check EVERY line of long documents):

1. CHECK FOR UNSHIFTED DATES: Look for dates in ANONYMIZED that are identical to ORIGINAL.
   All dates should be shifted by {time_offset} days.
   → Use shift_datetime(datetime_str, offset_days={time_offset}) to fix

2. CHECK FOR UNREDACTED PII: Look for PII that appears UNCHANGED:
   - Patient names, doctor names, staff names
   - Phone numbers, fax numbers, email addresses
   - Specific addresses or facility names
   → Use redact_text(text_to_redact="the PII text") to fix

IMPORTANT:
- Compare ORIGINAL vs ANONYMIZED to identify issues
- Only redact actual PII, not medical content
- Restore any medical terms that were incorrectly redacted
- Check the ENTIRE document, not just the first page

Call the appropriate tool for each issue you find. When done, summarize your findings.
""")

    # Image anonymization prompt (for PNG, JPG, DICOM images using Vision LLM + OCR)
    image_anonymization_prompt: str = field(default="""Analyze this medical image and identify ALL text that contains Personal Identifiable Information (PII) that should be redacted for patient privacy.

The following texts were detected in the image by OCR:
{ocr_text_list}

For EACH piece of PII you identify:
1. text: Copy the EXACT text as it appears (must match one of the OCR texts above as closely as possible)
2. type: Classify the PII type

PII categories to identify:
- name: Patient names, physician/doctor names, any person names
- date_of_birth: Dates of birth, DOB
- id_number: Patient IDs, medical record numbers (MRN), accession numbers, study IDs
- address: Physical addresses, street addresses
- location: Hospital names, clinic names, facility names, cities, institutions
- phone: Phone numbers, fax numbers
- email: Email addresses
- dates: Specific dates (admission date, study date, exam date, etc.) that could identify a patient

IMPORTANT:
- Focus on text that could identify a specific patient or person
- Include ALL identifying information, not just obvious names
- Medical record numbers, accession numbers, and study IDs are PII
- Hospital/facility names are PII (location)
- Dates associated with studies or admissions are PII
- Generic medical terms and measurements are NOT PII
""")

    # Image verification prompt (checks redacted images for remaining PII)
    image_verification_prompt: str = field(default="""Carefully analyze this medical image that has been redacted (black rectangles cover sensitive information).

Your task is to verify that ALL Personal Identifiable Information (PII) has been properly redacted.

Look for ANY remaining PII that is still visible and NOT covered by black redaction boxes:
- Patient names, physician names, any person names
- Dates of birth, birth dates
- Patient IDs, medical record numbers (MRN), accession numbers, study IDs
- Physical addresses, street addresses
- Hospital names, clinic names, facility names, institution names
- Phone numbers, fax numbers
- Email addresses
- Specific dates (admission dates, study dates, exam dates) that could identify a patient

IMPORTANT:
- Black rectangles are intentional redactions - ignore them
- Focus on ANY text that is STILL VISIBLE and contains PII
- Even partially visible PII should be reported
- If text is visible but clearly NOT PII (medical terms, measurements), do not report it

Set is_clean to TRUE only if you are confident that NO PII remains visible.
Set is_clean to FALSE if you find ANY remaining PII that needs additional redaction.

Be thorough - missing even one PII element is a privacy violation.
""")

    # PDF anonymization prompt (for classifying OCR-extracted text as PII)
    pdf_anonymization_prompt: str = field(default="""Analyze this medical document image and identify ALL text that contains Personal Identifiable Information (PII) that should be redacted for patient privacy.

The following texts were detected in the document by OCR:
{ocr_text_list}

For EACH piece of PII you identify:
1. text: Copy the EXACT text as it appears (must match one of the OCR texts above as closely as possible)
2. type: Classify the PII type

PII categories to identify:
- name: Patient names, physician/doctor names, any person names
- date_of_birth: Dates of birth, DOB
- id_number: Patient IDs, medical record numbers (MRN), accession numbers, study IDs
- address: Physical addresses, street addresses
- location: Hospital names, clinic names, facility names, cities, institutions
- phone: Phone numbers, fax numbers
- email: Email addresses
- dates: Specific dates (admission date, study date, exam date, etc.) that could identify a patient

IMPORTANT:
- Focus on text that could identify a specific patient or person
- Include ALL identifying information, not just obvious names
- Medical record numbers, accession numbers, and study IDs are PII
- Hospital/facility names are PII (location)
- Dates associated with studies or admissions are PII
- Generic medical terms and measurements are NOT PII
""")

    # PDF verification prompt (checks redacted PDF pages for remaining PII)
    pdf_verification_prompt: str = field(default="""Carefully analyze this medical document page that has been redacted (black rectangles cover sensitive information).

Your task is to verify that ALL Personal Identifiable Information (PII) has been properly redacted.

Look for ANY remaining PII that is still visible and NOT covered by black redaction boxes:
- Patient names, physician names, any person names
- Dates of birth, birth dates
- Patient IDs, medical record numbers (MRN), accession numbers, study IDs
- Physical addresses, street addresses
- Hospital names, clinic names, facility names, institution names
- Phone numbers, fax numbers
- Email addresses
- Specific dates (admission dates, study dates, exam dates) that could identify a patient

IMPORTANT:
- Black rectangles are intentional redactions - ignore them
- Focus on ANY text that is STILL VISIBLE and contains PII
- Even partially visible PII should be reported
- If text is visible but clearly NOT PII (medical terms, measurements), do not report it

Set is_clean to TRUE only if you are confident that NO PII remains visible.
Set is_clean to FALSE if you find ANY remaining PII that needs additional redaction.

Be thorough - missing even one PII element is a privacy violation.
""")

    # DICOM metadata anonymization prompt (for free-text DICOM tags)
    dicom_metadata_anonymization_prompt: str = field(default="""You are a PII anonymization agent. Analyze the following DICOM metadata tag values and redact ALL Personal Identifiable Information (PII).

These are free-text fields extracted from a DICOM medical image file header. They may contain embedded patient names, doctor names, hospital names, or other identifying information mixed with medical terminology.

=== DICOM METADATA TAGS ===
{tag_data}
=== END OF TAGS ===

You have the redact_text tool available. For EACH piece of PII you find, call:
  redact_text(text_to_redact="exact PII text")

PII categories to redact:
- Patient names, physician names, doctor names, staff names
- Hospital names, clinic names, institution names, facility names
- Physical addresses, street names
- Patient IDs, medical record numbers
- Phone numbers, fax numbers, email addresses
- Any other information that could identify a specific individual or institution

DO NOT redact:
- Medical terminology, diagnoses, procedures, medications
- Generic descriptions (e.g., "chest x-ray", "CT abdomen", "routine exam")
- Anatomical terms, body parts
- Imaging parameters or technical descriptions
- Generic locations like "EMERGENCY ROOM", "ICU", "OR"

IMPORTANT:
- Call redact_text for EACH piece of PII you find
- Use the EXACT text as it appears
- The tag name (before the colon) should NOT be redacted, only the value
- If a tag value contains no PII, skip it entirely
""")

    # Additional instructions that get appended to all prompts
    additional_instructions: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "column_detection_prompt": self.column_detection_prompt,
            "csv_anonymization_prompt": self.csv_anonymization_prompt,
            "text_anonymization_prompt": self.text_anonymization_prompt,
            "csv_verification_prompt": self.csv_verification_prompt,
            "text_verification_prompt": self.text_verification_prompt,
            "image_anonymization_prompt": self.image_anonymization_prompt,
            "image_verification_prompt": self.image_verification_prompt,
            "pdf_anonymization_prompt": self.pdf_anonymization_prompt,
            "pdf_verification_prompt": self.pdf_verification_prompt,
            "dicom_metadata_anonymization_prompt": self.dicom_metadata_anonymization_prompt,
            "additional_instructions": self.additional_instructions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptConfig":
        """Create from dictionary."""
        # Get defaults from a fresh instance
        defaults = cls()
        return cls(
            column_detection_prompt=data.get("column_detection_prompt", defaults.column_detection_prompt),
            csv_anonymization_prompt=data.get("csv_anonymization_prompt", defaults.csv_anonymization_prompt),
            text_anonymization_prompt=data.get("text_anonymization_prompt", defaults.text_anonymization_prompt),
            csv_verification_prompt=data.get("csv_verification_prompt", defaults.csv_verification_prompt),
            text_verification_prompt=data.get("text_verification_prompt", defaults.text_verification_prompt),
            image_anonymization_prompt=data.get("image_anonymization_prompt", defaults.image_anonymization_prompt),
            image_verification_prompt=data.get("image_verification_prompt", defaults.image_verification_prompt),
            pdf_anonymization_prompt=data.get("pdf_anonymization_prompt", defaults.pdf_anonymization_prompt),
            pdf_verification_prompt=data.get("pdf_verification_prompt", defaults.pdf_verification_prompt),
            dicom_metadata_anonymization_prompt=data.get("dicom_metadata_anonymization_prompt", defaults.dicom_metadata_anonymization_prompt),
            additional_instructions=data.get("additional_instructions", ""),
        )

    def get_column_detection_prompt(self, columns: str, sample_data: str) -> str:
        """Get formatted column detection prompt."""
        prompt = self.column_detection_prompt.format(
            columns=columns,
            sample_data=sample_data
        )
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_csv_anonymization_prompt(
        self,
        csv_data: str,
        skipped_columns: str = ""
    ) -> str:
        """Get formatted CSV anonymization prompt."""
        skipped_info = ""
        if skipped_columns:
            skipped_info = f"\n\nNOTE: The following columns have ALREADY been fully redacted and are not shown: {skipped_columns}"

        prompt = self.csv_anonymization_prompt.format(
            csv_data=csv_data,
            skipped_columns=skipped_info
        )
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_text_anonymization_prompt(self, text_data: str) -> str:
        """Get formatted text anonymization prompt."""
        prompt = self.text_anonymization_prompt.format(text_data=text_data)
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_csv_verification_prompt(
        self,
        comparison_data: str,
        time_offset: int
    ) -> str:
        """Get formatted CSV verification prompt."""
        prompt = self.csv_verification_prompt.format(
            comparison_data=comparison_data,
            time_offset=time_offset
        )
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_text_verification_prompt(
        self,
        original_text: str,
        anonymized_text: str,
        time_offset: int
    ) -> str:
        """Get formatted text verification prompt."""
        prompt = self.text_verification_prompt.format(
            original_text=original_text,
            anonymized_text=anonymized_text,
            time_offset=time_offset
        )
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_image_anonymization_prompt(self, ocr_text_list: str) -> str:
        """Get formatted image anonymization prompt."""
        prompt = self.image_anonymization_prompt.format(ocr_text_list=ocr_text_list)
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_image_verification_prompt(self) -> str:
        """Get formatted image verification prompt."""
        prompt = self.image_verification_prompt
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_pdf_anonymization_prompt(self, ocr_text_list: str) -> str:
        """Get formatted PDF anonymization prompt."""
        prompt = self.pdf_anonymization_prompt.format(ocr_text_list=ocr_text_list)
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_pdf_verification_prompt(self) -> str:
        """Get formatted PDF verification prompt."""
        prompt = self.pdf_verification_prompt
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt

    def get_dicom_metadata_anonymization_prompt(self, tag_data: str) -> str:
        """Get formatted DICOM metadata anonymization prompt."""
        prompt = self.dicom_metadata_anonymization_prompt.format(tag_data=tag_data)
        if self.additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{self.additional_instructions}"
        return prompt


# Default prompt configuration instance
DEFAULT_PROMPT_CONFIG = PromptConfig()


def get_prompt_descriptions() -> Dict[str, str]:
    """Get user-friendly descriptions for each prompt field for the UI."""
    return {
        "column_detection_prompt": "Scans CSV/Excel column headers to find columns that should be fully redacted (e.g. patient IDs, names). Runs before the main anonymization step.",
        "csv_anonymization_prompt": "Main anonymization prompt for CSV and Excel files. The AI uses this to find and redact PII in individual cells.",
        "text_anonymization_prompt": "Main anonymization prompt for plain text and Word (.docx) files. The AI uses this to find and redact PII in the document.",
        "csv_verification_prompt": "Quality check for CSV/Excel files. Compares the original with the anonymized version to catch missed PII and fix over-redaction.",
        "text_verification_prompt": "Quality check for text and Word files. Compares the original with the anonymized version to catch missed PII and fix over-redaction.",
        "image_anonymization_prompt": "Finds PII in images (PNG, JPG, DICOM) by combining Vision AI with OCR-detected text.",
        "image_verification_prompt": "Checks redacted images to verify that all PII has been properly covered by black boxes.",
        "pdf_anonymization_prompt": "Finds PII in PDF pages by combining Vision AI with OCR-detected text.",
        "pdf_verification_prompt": "Checks redacted PDF pages to verify that all PII has been properly covered by black boxes.",
        "dicom_metadata_anonymization_prompt": "Anonymizes free-text fields in DICOM file headers (e.g. physician names, hospital names embedded in metadata).",
        "additional_instructions": "Extra instructions appended to ALL prompts. Use this to add domain-specific rules or exceptions.",
    }


# Required template variables for each prompt field.
# These placeholders are filled in at runtime and MUST remain in the prompt text.
REQUIRED_TEMPLATE_VARIABLES: Dict[str, list] = {
    "column_detection_prompt": ["{columns}", "{sample_data}"],
    "csv_anonymization_prompt": ["{csv_data}", "{skipped_columns}"],
    "text_anonymization_prompt": ["{text_data}"],
    "csv_verification_prompt": ["{comparison_data}", "{time_offset}"],
    "text_verification_prompt": ["{original_text}", "{anonymized_text}", "{time_offset}"],
    "image_anonymization_prompt": ["{ocr_text_list}"],
    "image_verification_prompt": [],
    "pdf_anonymization_prompt": ["{ocr_text_list}"],
    "pdf_verification_prompt": [],
    "dicom_metadata_anonymization_prompt": ["{tag_data}"],
    "additional_instructions": [],
}


def get_template_variables() -> Dict[str, list]:
    """Get the required template variables for each prompt field."""
    return dict(REQUIRED_TEMPLATE_VARIABLES)


def validate_prompt_variables(field_name: str, prompt_text: str) -> list:
    """
    Validate that all required template variables are still present in the prompt.

    Returns a list of missing variable names (empty list if valid).
    """
    required = REQUIRED_TEMPLATE_VARIABLES.get(field_name, [])
    missing = [var for var in required if var not in prompt_text]
    return missing


def validate_all_prompts(prompts_dict: Dict[str, str]) -> Dict[str, list]:
    """
    Validate all prompts and return a dict of field_name -> missing variables.
    Only includes fields that have missing variables.
    """
    errors = {}
    for field_name, prompt_text in prompts_dict.items():
        missing = validate_prompt_variables(field_name, prompt_text)
        if missing:
            errors[field_name] = missing
    return errors
