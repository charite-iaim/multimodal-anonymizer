"""
Tools for agentic anonymization.
- shift_datetime: Shift dates/times by a specified offset
- redact_text: Replace PII text with asterisks
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TimeShiftInput(BaseModel):
    """Input schema for the time shift tool."""
    datetime_str: str = Field(description="The date/time string to shift (e.g., '2024-03-15', '15.03.2024', '03/15/2024 14:30')")
    offset_days: int = Field(description="Number of days to shift (positive = future, negative = past)")


class RedactTextInput(BaseModel):
    """Input schema for the redact text tool."""
    text_to_redact: str = Field(description="The exact PII text to redact (e.g., 'John Smith', '555-123-4567')")
    row_index: int = Field(default=-1, description="Row index (0-based) where the PII was found. Use -1 for plain text files.")
    column_name: str = Field(default="", description="Column name where the PII was found. Leave empty for plain text files.")


class RestoreTextInput(BaseModel):
    """Input schema for the restore text tool (to fix over-redaction)."""
    redacted_text: str = Field(description="The incorrectly redacted text (asterisks) to find")
    original_text: str = Field(description="The original text to restore (from the original data)")
    row_index: int = Field(default=-1, description="Row index (0-based) where the over-redaction occurred. Use -1 for plain text files.")
    column_name: str = Field(default="", description="Column name where the over-redaction occurred. Leave empty for plain text files.")


class RedactColumnInput(BaseModel):
    """Input schema for the redact column tool."""
    column_name: str = Field(description="The name of the column to redact entirely (e.g., 'subject_id', 'hadm_id')")
    reason: str = Field(description="Brief reason why this column contains PII (e.g., 'patient identifier', 'admission ID')")


# Special format markers for non-standard date formats
YEAR_ONLY_FORMAT = "YEAR_ONLY"  # e.g., "2140"
YEAR_RANGE_FORMAT = "YEAR_RANGE"  # e.g., "2011 - 2013"

# Common date/time formats to try when parsing
DATE_FORMATS = [
    # ISO formats
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",  # ISO with space separator and milliseconds
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    # ISO with 12-hour AM/PM formats (medical records often use these)
    "%Y-%m-%d %I:%M%p",
    "%Y-%m-%d %I:%M %p",
    "%Y-%m-%d %I:%M:%S%p",
    "%Y-%m-%d %I:%M:%S %p",
    # European formats
    "%d.%m.%Y",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    # US formats
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m-%d-%Y",
    # Text formats
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    # Time only
    "%H:%M:%S",
    "%H:%M",
]


def detect_format(datetime_str: str) -> Optional[str]:
    """
    Detect the format of a datetime string.

    Args:
        datetime_str: The datetime string to analyze

    Returns:
        The format string if detected, None otherwise
    """
    datetime_str = datetime_str.strip()

    # Check for year range format (e.g., "2011 - 2013", "2011-2013")
    year_range_match = re.match(r'^(\d{4})\s*[-–]\s*(\d{4})$', datetime_str)
    if year_range_match:
        return YEAR_RANGE_FORMAT

    # Check for year-only format (e.g., "2140")
    if re.match(r'^\d{4}$', datetime_str):
        return YEAR_ONLY_FORMAT

    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(datetime_str, fmt)
            return fmt
        except ValueError:
            continue

    return None


def shift_datetime_value(datetime_str: str, offset_days: int) -> str:
    """
    Shift a datetime string by the specified number of days.

    Args:
        datetime_str: The original datetime string
        offset_days: Number of days to shift

    Returns:
        The shifted datetime string in the same format
    """
    datetime_str = datetime_str.strip()
    detected_format = detect_format(datetime_str)

    if detected_format is None:
        # If we can't parse it, return original with a note
        return f"{datetime_str} [SHIFT_FAILED]"

    try:
        # Handle year-only format (e.g., "2140")
        if detected_format == YEAR_ONLY_FORMAT:
            year = int(datetime_str)
            # Convert days to years (approximate: 365.25 days per year)
            year_offset = offset_days // 365
            shifted_year = year + year_offset
            return str(shifted_year)

        # Handle year range format (e.g., "2011 - 2013")
        if detected_format == YEAR_RANGE_FORMAT:
            match = re.match(r'^(\d{4})\s*([-–])\s*(\d{4})$', datetime_str)
            if match:
                year1 = int(match.group(1))
                separator = match.group(2)
                year2 = int(match.group(3))
                year_offset = offset_days // 365
                shifted_year1 = year1 + year_offset
                shifted_year2 = year2 + year_offset
                # Preserve the original separator and spacing
                if ' - ' in datetime_str:
                    return f"{shifted_year1} - {shifted_year2}"
                elif ' – ' in datetime_str:
                    return f"{shifted_year1} – {shifted_year2}"
                else:
                    return f"{shifted_year1}{separator}{shifted_year2}"

        parsed = datetime.strptime(datetime_str, detected_format)
        shifted = parsed + timedelta(days=offset_days)
        return shifted.strftime(detected_format)
    except Exception:
        return f"{datetime_str} [SHIFT_FAILED]"


@tool("shift_datetime", args_schema=TimeShiftInput)
def shift_datetime(datetime_str: str, offset_days: int) -> str:
    """
    Shift a date or datetime by a specified number of days.

    Use this tool to anonymize dates by shifting them while preserving
    the relative time relationships in the data.

    Args:
        datetime_str: The date/time string to shift (e.g., '2024-03-15', '15.03.2024')
        offset_days: Number of days to shift (positive = future, negative = past)

    Returns:
        The shifted date/time string in the same format as the input
    """
    return shift_datetime_value(datetime_str, offset_days)


def find_and_shift_all_dates(text: str, offset_days: int) -> tuple[str, list[dict]]:
    """
    Find all dates in text and shift them.
    This is a helper function that can be used alongside the tool.

    Args:
        text: The text to process
        offset_days: Number of days to shift

    Returns:
        Tuple of (modified text, list of shifts made)
    """
    shifts = []

    # Patterns for common date formats
    patterns = [
        # ISO: 2024-03-15
        r'\b(\d{4}-\d{2}-\d{2})\b',
        # European: 15.03.2024
        r'\b(\d{2}\.\d{2}\.\d{4})\b',
        # Slash formats: 15/03/2024 or 03/15/2024
        r'\b(\d{2}/\d{2}/\d{4})\b',
        # With time: 2024-03-15 14:30:00
        r'\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\b',
    ]

    modified_text = text

    for pattern in patterns:
        matches = re.finditer(pattern, modified_text)
        for match in matches:
            original = match.group(1)
            shifted = shift_datetime_value(original, offset_days)

            if "[SHIFT_FAILED]" not in shifted:
                modified_text = modified_text.replace(original, shifted, 1)
                shifts.append({
                    "original": original,
                    "shifted": shifted,
                    "position": match.start()
                })

    return modified_text, shifts


@tool("redact_text", args_schema=RedactTextInput)
def redact_text(text_to_redact: str, row_index: int, column_name: str) -> str:
    """
    Redact PII text by replacing it with asterisks.

    Use this tool to anonymize personal identifiable information (PII) that was
    missed during the initial anonymization phase. The text will be replaced
    with asterisks of the same length.

    Args:
        text_to_redact: The exact PII text to redact (e.g., 'John Smith', '555-123-4567')
        row_index: Row index (0-based) where the PII was found
        column_name: Column name where the PII was found

    Returns:
        The redacted text (asterisks of the same length)
    """
    if not text_to_redact or not text_to_redact.strip():
        return "[REDACT_FAILED: empty text]"
    
    # Replace with asterisks of the same length
    redacted = "*" * len(text_to_redact)
    return redacted


def redact_text_value(text_to_redact: str) -> str:
    """
    Helper function to redact text without the tool wrapper.
    
    Args:
        text_to_redact: The text to redact
        
    Returns:
        Asterisks of the same length as the input
    """
    if not text_to_redact or not text_to_redact.strip():
        return ""
    return "*" * len(text_to_redact)


@tool("redact_column", args_schema=RedactColumnInput)
def redact_column(column_name: str, reason: str) -> str:
    """
    Mark an entire column for redaction because it contains PII.

    Use this tool when you identify that a column contains identifiers or other PII
    that should be redacted across ALL rows. This is more efficient than calling
    redact_text for each individual cell.

    Common columns to redact:
    - subject_id: Patient identifier
    - hadm_id: Hospital admission ID
    - stay_id: ICU stay identifier
    - note_id: Clinical note identifier
    - caregiver_id: Healthcare provider identifier
    - provider_id: Provider identifier
    - Any column containing names, IDs, or other identifiers

    Args:
        column_name: The exact name of the column to redact
        reason: Brief explanation of why this column contains PII

    Returns:
        Confirmation message with the column name and reason
    """
    if not column_name or not column_name.strip():
        return "[REDACT_COLUMN_FAILED: empty column name]"

    return f"[REDACT_COLUMN:{column_name}:{reason}]"


@tool("restore_text", args_schema=RestoreTextInput)
def restore_text(redacted_text: str, original_text: str, row_index: int, column_name: str) -> str:
    """
    Restore incorrectly redacted text (fix over-redaction).

    Use this tool when non-PII content was incorrectly redacted. This restores
    the original text that should not have been anonymized, such as:
    - Medical terminology (diabetes, hypertension, etc.)
    - Procedure names (colonoscopy, MRI, etc.)
    - Medication names (metformin, lisinopril, etc.)
    - Generic locations (EMERGENCY ROOM, ICU, HOME)

    Args:
        redacted_text: The incorrectly redacted text (asterisks) currently in the data
        original_text: The original text to restore (from the original data)
        row_index: Row index (0-based) where the over-redaction occurred
        column_name: Column name where the over-redaction occurred

    Returns:
        The original text that should be restored
    """
    if not original_text:
        return "[RESTORE_FAILED: no original text provided]"
    
    if not redacted_text:
        return "[RESTORE_FAILED: no redacted text to find]"
    
    return original_text
