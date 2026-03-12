"""
Tools for redacting PII in text and columns.
- redact_text: Replace PII text with asterisks
- redact_column: Mark entire columns for redaction
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class RedactTextInput(BaseModel):
    """Input schema for the redact text tool."""
    text_to_redact: str = Field(description="The exact PII text to redact (e.g., 'John Smith', '555-123-4567')")
    row_index: int = Field(default=-1, description="Row index (0-based) where the PII was found. Use -1 for plain text files.")
    column_name: str = Field(default="", description="Column name where the PII was found. Leave empty for plain text files.")


class RedactColumnInput(BaseModel):
    """Input schema for the redact column tool."""
    column_name: str = Field(description="The name of the column to redact entirely (e.g., 'subject_id', 'hadm_id')")
    reason: str = Field(description="Brief reason why this column contains PII (e.g., 'patient identifier', 'admission ID')")


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
