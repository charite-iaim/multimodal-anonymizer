"""
Text file processor using LLM for anonymization.
This processor anonymizes PII in plain text files like discharge summaries and clinical notes.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..llm_factory import create_chat_llm


class TextAnonymization(BaseModel):
    """Anonymization result for text content."""
    original_text: str = Field(description="Original text snippet containing PHI")
    anonymized_text: str = Field(description="Text with PHI replaced by asterisks")
    phi_category: str = Field(description="Category of PHI (name, date, address, id, etc.)")
    line_number: int = Field(description="Approximate line number where PHI was found")


class TextAnonymizationResult(BaseModel):
    """Result of text anonymization."""
    anonymized_content: str = Field(description="Complete anonymized text content")
    anonymizations: List[TextAnonymization] = Field(
        description="List of PHI items that were anonymized"
    )


class TextProcessor(FileProcessor):
    """Processor for plain text files using LLM for PII detection and anonymization."""

    def __init__(self, config: AnonymizerConfig):
        """Initialize text processor."""
        super().__init__(config)

        # Initialize LLM for PII detection and anonymization
        self.llm = create_chat_llm(
            config=config,
            structured_output=TextAnonymizationResult,
        )

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a text file (.txt or .hea ECG header)."""
        return file_path.suffix.lower() in [".txt", ".hea"]

    def extract_content(self, file_path: Path) -> str:
        """Extract text content as string."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize text file using LLM.

        Steps:
        1. Read text file
        2. Use LLM to identify and redact PII
        3. Save anonymized text with same structure
        4. Save JSON with anonymization details

        Args:
            input_path: Path to input text file
            output_path: Path to save anonymized text file
        """
        print(f"Processing: {input_path.name}")

        # Step 1: Read text file
        content = self.extract_content(input_path)
        print(f"Read {len(content)} characters")

        if not content.strip():
            print("Empty text file, saving as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return

        # Step 2: Anonymize using LLM
        print("Anonymizing PII using LLM...")
        anonymization_result = self._anonymize_with_llm(content)
        print(f"Anonymized {len(anonymization_result.anonymizations)} PHI items")

        # Step 3: Save anonymized text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(anonymization_result.anonymized_content)
        print(f"Saved anonymized text to: {output_path}")

        # Step 4: Save JSON with anonymization details (only if debug mode is enabled)
        if self.config.save_debug_files:
            json_output_path = output_path.with_suffix('.json')
            self._save_json_output(anonymization_result, input_path, output_path, json_output_path)
            print(f"Saved anonymization details to: {json_output_path}")

    def _anonymize_with_llm(self, content: str) -> TextAnonymizationResult:
        """
        Use LLM to identify and anonymize PII in text content.

        Args:
            content: Text content to anonymize

        Returns:
            TextAnonymizationResult with anonymized content and details
        """
        # Limit content size for context window (keep first 10000 characters)
        max_chars = 10000
        content_to_process = content[:max_chars]
        was_truncated = len(content) > max_chars

        prompt = f"""Analyze this medical document to anonymize all Personal Identifiable Information (PII).

    PII categories to redact:
    - name: Patient names, physician names, doctor names, family member names, caregiver names
    - date: Dates (dates of birth, admission dates, discharge dates, specific dates in any format)
    - address: Physical addresses, street addresses, facility names, hospital names, location names
    - id: Patient IDs, medical record numbers, unit numbers, subject IDs, any numeric identifiers
    - phone: Phone numbers
    - fax: Fax numbers
    - email: Email addresses

    Instructions:
    1. Read through the entire document carefully
    2. Identify all instances of PHI in the categories above
    3. Replace EVERY CHARACTER of each PHI instance (including spaces, hyphens, slashes, colons, etc.) with an asterisk (*)
    4. Return the COMPLETE anonymized document with all PHI replaced
    5. Keep all other content (medical terminology, procedures, medications, etc.) unchanged
    6. Preserve the original document structure and formatting
    7. Keep times, measurements, and medical data intact
    8. Keep Metadata and file-specific information intact (e.g., ECG header info)

    Redaction examples:
    - "John Doe" → "********" (8 asterisks)
    - "May 10, 2024" → "************" (12 asterisks)
    - "23/09/2140" → "**********" (10 asterisks)
    - "123456789" → "*********" (9 asterisks)
    - "Dr. Jane Smith" → "**************" (14 asterisks)
    - "General Hospital" → "****************" (16 asterisks)
    - "<PER>47646408</PER>" → "<PER>********</PER>" (tags preserved, content replaced)
    - "<subject_id>: 10005749" → "<subject_id>: ********" (tag and colon preserved)
    - "45790175.dat 16 200.0(0)/mV 16 0 19 3475 0 I" → "********.dat 16 200.0(0)/mV 16 0 19 3475 0 I" (metadata preserved, ID replaced)

    CRITICAL:
    - Replace EVERY character (including spaces, hyphens, slashes, colons, etc.) in PHI with an asterisk
    - For XML-like tags containing PHI, preserve the tags but replace the content
    - Return the COMPLETE document with ALL PHI replaced
    - Preserve all line breaks, formatting, and structure
    - Only replace PHI, not medical information or general terms

    Provide:
    1. anonymized_content: The complete anonymized text with all PHI replaced
    2. anonymizations: List of specific PHI items that were redacted, with:
    - original_text: The original PHI text
    - anonymized_text: The asterisk replacement
    - phi_category: Category (name, date, address, id, phone, fax, email)
    - line_number: Approximate line number (best estimate)

    === DOCUMENT TO ANONYMIZE ===
    {content_to_process}
    """

        message = HumanMessage(content=prompt)

        try:
            result: TextAnonymizationResult = self.llm.invoke([message])

            # Print details
            print(f"\nAnonymization details:")
            for anon in result.anonymizations:
                print(f"  Line ~{anon.line_number}: {anon.phi_category} - '{anon.original_text}' → '{anon.anonymized_text}'")

            return result

        except Exception as e:
            print(f"Error during anonymization: {e}")
            import traceback
            traceback.print_exc()
            # Return original content if anonymization fails
            return TextAnonymizationResult(
                anonymized_content=content,
                anonymizations=[]
            )

    def _save_json_output(
        self,
        result: TextAnonymizationResult,
        input_path: Path,
        output_path: Path,
        json_output_path: Path
    ) -> None:
        """
        Save anonymization details as JSON.

        Args:
            result: TextAnonymizationResult object
            input_path: Path to original input file
            output_path: Path to anonymized output file
            json_output_path: Path to save JSON output
        """
        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "llm_text_anonymization_asterisk",
                "total_phi_items": len(result.anonymizations)
            },
            "anonymizations": [
                {
                    "line_number": anon.line_number,
                    "phi_category": anon.phi_category,
                    "original_text": anon.original_text,
                    "anonymized_text": anon.anonymized_text
                }
                for anon in result.anonymizations
            ]
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
