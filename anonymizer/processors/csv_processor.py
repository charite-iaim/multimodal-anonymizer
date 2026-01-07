"""
CSV file processor using LLM for anonymization.
This processor anonymizes PII (names, dates, addresses, IDs) in CSV files.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig


class PHISpan(BaseModel):
    """A span of PHI text to be redacted."""
    text: str = Field(description="The exact PHI text to redact")
    type: str = Field(description="Type of PII (name, date, address, id, phone, email)")


class FieldAnonymization(BaseModel):
    """Anonymization result for a single cell."""
    row_index: int = Field(description="Row index (0-based, excluding header)")
    column_name: str = Field(description="Column name")
    phi_spans: List[PHISpan] = Field(description="List of PHI text spans found in this cell")


class CSVAnonymizationResult(BaseModel):
    """Result of CSV anonymization."""
    anonymizations: List[FieldAnonymization] = Field(
        description="List of cells that were anonymized"
    )


class CSVProcessor(FileProcessor):
    """Processor for CSV files using LLM for PII detection and anonymization."""

    def __init__(self, config: AnonymizerConfig):
        """Initialize CSV processor."""
        super().__init__(config)

        # Initialize LLM for PII detection and anonymization
        self.llm = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
        ).with_structured_output(CSVAnonymizationResult)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a CSV."""
        return file_path.suffix.lower() == ".csv"

    def extract_content(self, file_path: Path) -> str:
        """Extract CSV content as string."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize CSV file using LLM.

        Steps:
        1. Read CSV file
        2. Use LLM to identify and redact PII in all cells
        3. Save anonymized CSV with same structure
        4. Save JSON with anonymization details

        Args:
            input_path: Path to input CSV
            output_path: Path to save anonymized CSV
        """
        print(f"Processing: {input_path.name}")

        # Step 1: Read CSV file
        rows, headers = self._read_csv(input_path)
        print(f"Found {len(rows)} rows with columns: {', '.join(headers)}")

        if not rows:
            print("Empty CSV file, saving as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path, 'r', encoding='utf-8') as src:
                with open(output_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            return

        # Step 2: Anonymize using LLM
        print("Anonymizing PII using LLM...")
        anonymization_result = self._anonymize_with_llm(rows, headers)
        print(f"Anonymized {len(anonymization_result.anonymizations)} cells")

        # Step 3: Apply anonymizations
        anonymized_rows = self._apply_anonymizations(rows, anonymization_result)

        # Step 4: Save anonymized CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_csv(output_path, headers, anonymized_rows)
        print(f"Saved anonymized CSV to: {output_path}")

        # Step 5: Save JSON with anonymization details
        json_output_path = output_path.with_suffix('.json')
        self._save_json_output(anonymization_result, input_path, output_path, json_output_path)
        print(f"Saved anonymization details to: {json_output_path}")

    def _read_csv(self, file_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
        """
        Read CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (rows as list of dicts, header list)
        """
        rows = []
        headers = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        return rows, headers

    def _write_csv(self, file_path: Path, headers: List[str], rows: List[Dict[str, str]]) -> None:
        """
        Write CSV file.

        Args:
            file_path: Path to save CSV
            headers: Column headers
            rows: Rows as list of dicts
        """
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def _anonymize_with_llm(
        self,
        rows: List[Dict[str, str]],
        headers: List[str]
    ) -> CSVAnonymizationResult:
        """
        Use LLM to identify and anonymize PII in CSV data.

        Args:
            rows: CSV rows as list of dicts
            headers: Column headers

        Returns:
            CSVAnonymizationResult with anonymization details
        """
        # Prepare CSV preview for LLM (limit to manageable size)
        max_rows_to_send = min(len(rows), 50)  # Limit for context window

        # Create a formatted representation of the CSV
        csv_preview = self._format_csv_for_llm(rows[:max_rows_to_send], headers)

        prompt = f"""Analyze this CSV data from a medical discharge note and identify all Personal Identifiable Information (PII) that needs to be redacted.

CSV Data (showing {max_rows_to_send} of {len(rows)} rows):
{csv_preview}

PII categories to identify:
- name: Patient names, physician names, doctor names, family member names
- date: Dates (dates of birth, admission dates, discharge dates, specific dates in format YYYY-MM-DD or similar)
- ages: Patient ages
- address: Physical addresses, street addresses, facility names, location names
- id: Patient IDs, medical record numbers, unit numbers, any numeric identifiers
- phone: Phone numbers
- fax: Fax numbers
- email: Email addresses

Instructions:
1. Examine each cell in the CSV data
2. Identify ALL specific PHI text spans in each cell
3. For each PHI text span, provide the EXACT text as it appears

For each cell containing PHI, provide:
- row_index: The row number (0-based, excluding header row)
- column_name: The column name
- phi_spans: List of PHI text spans found, where each span contains:
  - text: The EXACT PHI text to redact (e.g., "Emily Carter", "2140-09-28", "5837209")
  - type: The PII category (name, date, address, id, phone, email, ages)

CRITICAL:
- Provide the EXACT text for each PHI span as it appears in the cell
- Include ALL PHI occurrences in each cell
- Be precise with the text matching
- Only include cells that actually contain PHI (do not include cells with general medical information, medications, procedures, etc.)

Examples:
- If you see "Name: Emily Carter", the phi_span text should be "Emily Carter" (not "Name: Emily Carter")
- If you see "DOB: 2140-09-28", the phi_span text should be "2140-09-28"
- If you see "ID: 5837209", the phi_span text should be "5837209"
"""

        message = HumanMessage(content=prompt)

        try:
            result: CSVAnonymizationResult = self.llm.invoke([message])

            # Print details
            for anon in result.anonymizations:
                phi_types = list(set([span.type for span in anon.phi_spans]))
                print(f"  Row {anon.row_index}, Column '{anon.column_name}': {phi_types}")
                for span in anon.phi_spans:
                    print(f"    {span.type}: '{span.text}'")

            return result

        except Exception as e:
            print(f"Error during anonymization: {e}")
            import traceback
            traceback.print_exc()
            return CSVAnonymizationResult(anonymizations=[])

    def _format_csv_for_llm(self, rows: List[Dict[str, str]], headers: List[str]) -> str:
        """
        Format CSV data for LLM prompt.

        Args:
            rows: CSV rows
            headers: Column headers

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("Headers: " + " | ".join(headers))
        lines.append("-" * 80)

        for idx, row in enumerate(rows):
            lines.append(f"Row {idx}:")
            for header in headers:
                value = row.get(header, "")
                lines.append(f"  {header}: {value}")
            lines.append("")

        return "\n".join(lines)

    def _apply_anonymizations(
        self,
        rows: List[Dict[str, str]],
        result: CSVAnonymizationResult
    ) -> List[Dict[str, str]]:
        """
        Apply anonymizations to CSV rows by replacing PHI with asterisks.

        Args:
            rows: Original CSV rows
            result: Anonymization result from LLM

        Returns:
            Anonymized rows
        """
        # Create a copy of rows
        anonymized_rows = [row.copy() for row in rows]

        # Apply each anonymization
        for anon in result.anonymizations:
            if 0 <= anon.row_index < len(anonymized_rows):
                if anon.column_name in anonymized_rows[anon.row_index]:
                    original_text = anonymized_rows[anon.row_index][anon.column_name]
                    anonymized_text = original_text

                    # Sort PHI spans by their position in the text (reverse order to maintain indices)
                    # We need to sort by position to replace from end to beginning
                    spans_with_pos = []
                    for span in anon.phi_spans:
                        pos = anonymized_text.find(span.text)
                        if pos != -1:
                            spans_with_pos.append((pos, span))

                    # Sort in reverse order by position
                    spans_with_pos.sort(key=lambda x: x[0], reverse=True)

                    # Replace each PHI span with asterisks
                    for pos, span in spans_with_pos:
                        # Replace the PHI text with asterisks (one asterisk per character)
                        replacement = '*' * len(span.text)
                        anonymized_text = (
                            anonymized_text[:pos] +
                            replacement +
                            anonymized_text[pos + len(span.text):]
                        )

                    anonymized_rows[anon.row_index][anon.column_name] = anonymized_text
                    print(f"  Applied: Row {anon.row_index}, Column '{anon.column_name}'")

        return anonymized_rows

    def _save_json_output(
        self,
        result: CSVAnonymizationResult,
        input_path: Path,
        output_path: Path,
        json_output_path: Path
    ) -> None:
        """
        Save anonymization details as JSON.

        Args:
            result: CSVAnonymizationResult object
            input_path: Path to original input file
            output_path: Path to anonymized output file
            json_output_path: Path to save JSON output
        """
        # Count total PHI spans
        total_phi_spans = sum(len(anon.phi_spans) for anon in result.anonymizations)

        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "llm_csv_anonymization_asterisk",
                "total_cells_with_phi": len(result.anonymizations),
                "total_phi_spans": total_phi_spans
            },
            "anonymizations": [
                {
                    "row_index": anon.row_index,
                    "column_name": anon.column_name,
                    "phi_spans": [
                        {
                            "text": span.text,
                            "type": span.type,
                            "redacted_as": '*' * len(span.text)
                        }
                        for span in anon.phi_spans
                    ]
                }
                for anon in result.anonymizations
            ]
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
