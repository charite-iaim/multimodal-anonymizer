"""
Custom prompt configuration.

This module contains a basic structure for defining custom prompts for the anonymization process. 
It can be used as a template to create specific prompts for different datasets or use cases.

Use with: python anonymize_agentic.py --prompt-config custom <input>
"""

from dataclasses import dataclass, field
from anonymizer.prompt_config import PromptConfig


@dataclass
class CustomPromptConfig(PromptConfig):
    """
    Custom prompt configuration.

    In this file it is possible to define custom prompts for different phases of the anonymization process.
    """

    # Phase 1b: Column-level PII detection for tabular data
    column_detection_prompt: str = field(default="""You are a PII detection agent. Analyze this CSV structure and identify columns that contain Personal Identifiable Information (PII) that should be ENTIRELY redacted.

        CSV Column Headers: {columns}

        Sample Data (first rows):
        {sample_data}

        Your task: Identify columns where ALL values should be redacted because the column contains ONLY identifiers or PII.

        Common PII columns to look for:
        ---- LIST COLUMNS WITH PII HERE ----

        DO NOT mark as PII columns:
        ---- LIST COLUMNS THAT LOOK LIKE PII BUT ARE NOT (e.g. generic medical terms, lab values, etc.) ----
                                                
        For EACH column that should be entirely redacted, call:
        redact_column(column_name="exact_column_name", reason="brief reason")

        If no columns need full redaction, just respond with "No columns identified for full redaction."
        """)

    # Phase 2: PII anonymization for text files
    text_anonymization_prompt: str = field(default="""You are a PII anonymization agent for medical data. Analyze this text and redact ALL Personal Identifiable Information (PII).

        CRITICAL: You MUST scan the ENTIRE document, even if it's very long. Do NOT skip any sections!

        IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

        === TEXT TO ANALYZE ===
        {text_data}
        === END OF TEXT ===

        You have the redact_text tool available. For EACH piece of PII you find, call:
        redact_text(text_to_redact="exact PII text") e.g. redact_text(text_to_redact="John Doe")

        PII categories to redact:
        ---- LIST PII CATEGORIES AND EXAMPLES HERE (e.g. name: patient names, doctor names; id: patient_id, insurance_id, etc.) ----

        Example reductions for .txt files:
        ---- HERE PROVIDE EXAMPLES OF PII REDACTION IN TEXT FILES (e.g. "The patient Alice Bob is admitted." -> "The patient ********** is admitted.") ----                                  

        DO NOT redact:
        ---- LIST TYPES OF INFORMATION THAT SHOULD NOT BE REDACTED (e.g. medical terms, generic locations, lab values, etc.) ----

        IMPORTANT:
        - Call redact_text for EACH piece of PII you find
        - Use the EXACT text as it appears (preserve spacing and punctuation)
        - Redact complete names, not just first or last names
        - Be thorough - check for all PII types
        """)

    # Phase 2: PII anonymization for tabular data
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
        ---- LIST PII CATEGORIES AND EXAMPLES HERE (e.g. name: patient names, doctor names; id: patient_id, insurance_id, etc.) ----

        EFFICIENCY TIP: If you notice a column like "patient_id" contains identifiers in every row,
        use redact_column ONCE instead of calling redact_text for each row!

        DO NOT redact:
        ---- LIST TYPES OF INFORMATION THAT SHOULD NOT BE REDACTED (e.g. medical terms, generic locations, lab values, etc.) ----

        IMPORTANT:
        - Call redact_text for EACH piece of PII you find
        - Use the EXACT text as it appears (preserve spacing)
        - Use absolute row indices (0-based)
        - Redact the specific PII text, not the entire cell
        """)

    # Image anonymization prompt
    image_anonymization_prompt: str = field(default="""Analyze this medical image and identify ALL text that contains Personal Identifiable Information (PII) that should be redacted for patient privacy.

        The following texts were detected in the image by OCR:
        {ocr_text_list}

        For EACH piece of PII you identify:
        1. text: Copy the EXACT text as it appears (must match one of the OCR texts above as closely as possible)
        2. type: Classify the PII type

        PII categories to identify:
        ---- LIST PII CATEGORIES AND EXAMPLES HERE (e.g. name: patient names, doctor names; id: patient_id, insurance_id, etc.) ----

        IMPORTANT:
        - Focus on text that could identify a specific patient or person
        - Include ALL identifying information, not just obvious names
        - Hospital/facility names are PII (location)
        - Dates associated with studies or admissions are PII
        - Generic medical terms, measurements are NOT PII
        """)

    # Image verification prompt
    image_verification_prompt: str = field(default="""Carefully analyze this medical image that has been redacted (black rectangles cover sensitive information).

        Your task is to verify that ALL Personal Identifiable Information (PII) has been properly redacted.

        Look for ANY remaining PII that is still visible and NOT covered by black redaction boxes:
        ---- LIST PII CATEGORIES AND EXAMPLES HERE (e.g. name: patient names, doctor names; id: patient_id, insurance_id, etc.) ----

        IMPORTANT:
        - Black rectangles are intentional redactions - ignore them
        - Focus on ANY text that is STILL VISIBLE and contains PII
        - Even partially visible PII should be reported
        - If text is visible but clearly NOT PII (e.g. medical terms, measurements), do not report it

        Set is_clean to TRUE only if you are confident that NO PII remains visible.
        Set is_clean to FALSE if you find ANY remaining PII that needs additional redaction.

        Be thorough - missing even one PII element is a privacy violation.
        """)

    # Phase 3: Text verification prompt
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

        Your tasks (BE THOROUGH - check EVERY line):

        1. CHECK FOR UNSHIFTED DATES: Look for dates in ANONYMIZED that are identical to ORIGINAL.
        All dates should be shifted by {time_offset} days.
        → Use shift_datetime(datetime_str, offset_days={time_offset}) to fix

        2. CHECK FOR UNREDACTED PII: Look for PII that appears UNCHANGED:
        --- LIST PII CATEGORIES AND EXAMPLES HERE (e.g. name: patient names, doctor names; id: patient_id, insurance_id, etc.) ----
        → Use redact_text(text_to_redact="the PII text") to fix

        IMPORTANT:
        - Compare ORIGINAL vs ANONYMIZED to identify issues
        - Only redact actual PII, not medical content
        - Check the ENTIRE document, not just the first page

        Call the appropriate tool for each issue you find. When done, summarize your findings.
        """)

    # PDF anonymization prompt
    pdf_anonymization_prompt: str = field(default="""Analyze the following texts extracted from a medical document and identify which ones contain Personal Identifiable Information (PII).

        Texts found in the document:
        {text_list}

        PII categories to identify:
        ---- LIST PII CATEGORIES AND EXAMPLES HERE (e.g. name: patient names, doctor names; id: patient_id, insurance_id, etc.) ----

        For each text that contains PII, provide:
        1. text: The EXACT text as it appears above (must match exactly)
        2. type: The PII category

        Only include texts that actually contain PII. If a text is just a medical term, or general information, do not include it.
        """)


# Create default Custom prompt config instance
CUSTOM_PROMPT_CONFIG = CustomPromptConfig()
