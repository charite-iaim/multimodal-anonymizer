"""
CSV file processor using LLM for anonymization.
This processor anonymizes PII (names, dates, addresses, IDs) in CSV files.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig


class FieldAnonymization(BaseModel):
    """Anonymization result for a single cell."""
    row_index: int = Field(description="Row index (0-based, excluding header)")
    column_name: str = Field(description="Column name")
    anonymized_value: str = Field(description="The complete cell content with all PHI replaced by asterisks")


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
        # Set a timeout of 5 minutes for API requests
        self.llm = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
            timeout=300,  # 5 minutes timeout
            request_timeout=300,  # Also set request_timeout
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

        # Step 5: Save JSON with anonymization details (only if debug mode is enabled)
        if self.config.save_debug_files:
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
            headers = [h for h in (reader.fieldnames or []) if h is not None]
            # Filter out None keys that can appear from malformed CSVs with extra delimiters
            rows = [{k: v for k, v in row.items() if k is not None} for row in reader]

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
        Processes all rows in batches to handle large CSV files.

        Args:
            rows: CSV rows as list of dicts
            headers: Column headers

        Returns:
            CSVAnonymizationResult with anonymization details
        """
        # Process in batches to handle large CSVs
        batch_size = 50
        all_anonymizations = []

        total_batches = (len(rows) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(rows))
            batch_rows = rows[start_idx:end_idx]

            print(f"  Processing batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx-1})")

            # Create a formatted representation of the CSV batch
            csv_preview = self._format_csv_for_llm(batch_rows, headers, start_idx)

            prompt = f"""Analyze this CSV data from a medical discharge note and anonymize all Personal Identifiable Information (PII) by replacing each PHI character with an asterisk (*).

CSV Data (rows {start_idx} to {end_idx-1} of {len(rows)} total rows):
{csv_preview}

PII categories to redact:
- name: Patient names, physician names, doctor names, family member names
- date: Dates (dates of birth, admission dates, discharge dates, specific dates in format YYYY-MM-DD or similar)
- ages: Patient ages (e.g., "64")
- address: Physical addresses, street addresses, facility names, location names (e.g. hospital names)
- id: Patient IDs, medical record numbers, unit numbers, any numeric identifiers
- phone: Phone numbers
- fax: Fax numbers
- email: Email addresses

Instructions:
1. Examine each cell in the CSV data
2. For each cell containing PHI, replace EVERY CHARACTER of the PHI text (including spaces) with an asterisk (*)
3. Return the complete cell content with PHI replaced by asterisks
4. Keep all other content unchanged
5. Make sure to only redact actual PHI, do not redact medical terms, procedures, medications, diagnosis or other non-PII content

Redaction examples:
- "Emily Carter" → "************" (12 asterisks)
- "2140-09-28" → "**********" (10 asterisks)
- "64" → "**" (2 asterisks)
- "617-555-3942" → "************" (12 asterisks)
- "discharge_location: DIRECT EMER." → "discharge_location: DIRECT EMER." (no change, as "DIRECT EMER." is not PHI)
- "admission_location: CLINIC REFERRAL" → "admission_location: CLINIC REFERRAL" (no change, as "CLINIC REFERRAL" is not PHI)
- "poe_seq: 5" → "poe_seq: 5" (no change, as "5" is not PHI)

For each cell containing PHI, provide:
- row_index: The row number (0-based, excluding header row - use the absolute row index from the entire CSV)
- column_name: The column name
- anonymized_value: The COMPLETE cell content with ALL PHI replaced by asterisks

CRITICAL:
- Replace EVERY character (including spaces, hyphens, etc.) in PHI with an asterisk
- Return the COMPLETE cell content, not truncated
- Only include cells that contain PHI (skip cells with only medical info, medications, procedures, etc.)
- Preserve all non-PHI text exactly as it appears
- Use the absolute row_index from the entire CSV file (not relative to this batch)

Example:
Original: "Name:  Emily Carter                     Unit No:   5837209"
Anonymized: "Name:  ************                     Unit No:   *******"
"""

            message = HumanMessage(content=prompt)

            # Retry logic: try up to 3 times
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    if retry_count > 0:
                        print(f"    Retry {retry_count}/{max_retries - 1}")

                    print(f"    Sending request to Azure OpenAI... (timeout: 5min)")
                    start_time = time.time()
                    result: CSVAnonymizationResult = self.llm.invoke([message])
                    elapsed_time = time.time() - start_time
                    print(f"    Response received in {elapsed_time:.1f}s")

                    # Add batch results to overall results
                    all_anonymizations.extend(result.anonymizations)

                    # Print details for this batch
                    for anon in result.anonymizations:
                        print(f"    Row {anon.row_index}, Column '{anon.column_name}'")

                    success = True

                except Exception as e:
                    retry_count += 1
                    elapsed_time = time.time() - start_time
                    print(f"  Error during batch {batch_num + 1} anonymization after {elapsed_time:.1f}s: {e}")

                    if retry_count < max_retries:
                        wait_time = retry_count * 5  # Exponential backoff: 5s, 10s, 15s
                        print(f"  Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"  Failed after {max_retries} attempts. Moving to next batch.")
                        import traceback
                        traceback.print_exc()

        return CSVAnonymizationResult(anonymizations=all_anonymizations)

    def _format_csv_for_llm(self, rows: List[Dict[str, str]], headers: List[str], start_idx: int = 0) -> str:
        """
        Format CSV data for LLM prompt.

        Args:
            rows: CSV rows
            headers: Column headers
            start_idx: Starting index for row numbering (for batch processing)

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("Column Headers:")
        lines.append(", ".join(headers))
        lines.append("")

        for idx, row in enumerate(rows):
            absolute_idx = start_idx + idx
            lines.append(f"=== ROW {absolute_idx} ===")
            for header in headers:
                value = row.get(header, "")
                lines.append(f"{header}: {value}")
            lines.append("")

        return "\n".join(lines)

    def _apply_anonymizations(
        self,
        rows: List[Dict[str, str]],
        result: CSVAnonymizationResult
    ) -> List[Dict[str, str]]:
        """
        Apply anonymizations to CSV rows using LLM-generated anonymized text.

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
                    # Use the anonymized value provided by the LLM
                    anonymized_rows[anon.row_index][anon.column_name] = anon.anonymized_value
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
        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "llm_csv_anonymization_asterisk",
                "total_cells_anonymized": len(result.anonymizations)
            },
            "anonymizations": [
                {
                    "row_index": anon.row_index,
                    "column_name": anon.column_name,
                    "anonymized_value_preview": anon.anonymized_value[:200] + "..." if len(anon.anonymized_value) > 200 else anon.anonymized_value
                }
                for anon in result.anonymizations
            ]
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
