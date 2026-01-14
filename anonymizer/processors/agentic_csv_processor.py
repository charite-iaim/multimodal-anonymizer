"""
Agentic CSV file processor using LLM with tool-calling for anonymization.

This processor uses a two-step agentic approach:
1. LLM identifies and shifts dates/times using the shift_datetime tool
2. LLM anonymizes all other PII (but not the already-shifted dates)
"""

import json
import csv
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
import random
import time

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..tools.time_shift_tool import shift_datetime, redact_text, restore_text


class DateTimeShift(BaseModel):
    """A date/time shift in the CSV."""
    row_index: int = Field(description="Row index (0-based)")
    column_name: str = Field(description="Column name")
    original_value: str = Field(description="Original date/time value")
    shifted_value: str = Field(description="Shifted date/time value")


class VerificationIssue(BaseModel):
    """An issue found during verification."""
    issue_type: str = Field(description="Type: 'unshifted_date', 'unredacted_pii', 'over_redaction'")
    row_index: int = Field(description="Row index (0-based)")
    column_name: str = Field(description="Column name")
    description: str = Field(description="Description of the issue")
    original_text: str = Field(description="The problematic text")
    suggested_fix: str = Field(description="Suggested replacement text")


class VerificationResult(BaseModel):
    """Result of verification."""
    issues: List[VerificationIssue] = Field(description="List of issues found")
    summary: str = Field(description="Brief summary of verification")


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


class AgenticCSVProcessor(FileProcessor):
    """
    Agentic processor for CSV files using LLM with tool-calling.

    This processor implements a two-phase approach:
    1. Time-Shift Phase: LLM uses the shift_datetime tool to find and shift all dates
    2. Anonymization Phase: LLM anonymizes all other PII (names, addresses, IDs, etc.)
    """

    def __init__(self, config: AnonymizerConfig, time_offset_days: Optional[int] = None):
        """
        Initialize agentic CSV processor.

        Args:
            config: Configuration object with LLM settings
            time_offset_days: Fixed offset for time shifting. If None, a random offset is generated.
        """
        super().__init__(config)

        # Generate random offset if not provided (between -365 and +365 days)
        if time_offset_days is None:
            self.time_offset_days = random.randint(-365, 365)
        else:
            self.time_offset_days = time_offset_days

        # Initialize LLM with tools for phase 1 (time shifting)
        self.llm_with_tools = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
            timeout=300,
        ).bind_tools([shift_datetime])

        # Initialize LLM with tools for phase 2 (PII anonymization)
        self.llm_anonymize = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
            timeout=300,
        ).bind_tools([redact_text])

        # Initialize LLM with tools for phase 3 (verification)
        self.llm_verify = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
            timeout=300,
        ).bind_tools([shift_datetime, redact_text, restore_text])

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a CSV."""
        return file_path.suffix.lower() == ".csv"

    def extract_content(self, file_path: Path) -> str:
        """Extract CSV content as string."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def anonymize(self, input_path: Path, output_path: Path, verify: bool = True) -> None:
        """
        Anonymize CSV file using agentic LLM approach.

        Steps:
        1. Read CSV file
        2. Phase 1: Extract and shift all dates/times using regex
        3. Phase 2: Use LLM to anonymize all other PII
        4. Phase 3: Verification agent checks and fixes any issues
        5. Save anonymized CSV

        Args:
            input_path: Path to input CSV
            output_path: Path to save anonymized CSV
            verify: Whether to run the verification phase (default: True)
        """
        # Convert to Path if string
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        output_path = Path(output_path) if isinstance(output_path, str) else output_path

        print(f"Processing (Agentic): {input_path.name}")
        print(f"Time offset: {self.time_offset_days} days")

        # Step 1: Read CSV file
        rows, headers = self._read_csv(input_path)
        original_rows = [row.copy() for row in rows]  # Keep original for verification
        print(f"Found {len(rows)} rows with columns: {', '.join(headers)}")

        if not rows:
            print("Empty CSV file, saving as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path, 'r', encoding='utf-8') as src:
                with open(output_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            return

        # Step 2: Phase 1 - Time shifting with regex extraction
        print("\n=== Phase 1: Time Shifting ===")
        shifted_rows, dates_shifted = self._phase1_shift_times(rows, headers)
        print(f"Shifted {len(dates_shifted)} date/time values")

        # Step 3: Phase 2 - Anonymize other PII (agentic with redact_text tool)
        print("\n=== Phase 2: PII Anonymization ===")
        anonymized_rows, pii_redactions = self._phase2_anonymize_pii(shifted_rows, headers)
        print(f"Applied {pii_redactions} PII redactions")

        # Step 5: Phase 3 - Verification (optional but recommended)
        if verify:
            print("\n=== Phase 3: Verification ===")
            anonymized_rows, fixes_applied = self._phase3_verify_and_fix(
                original_rows, anonymized_rows, headers
            )
            print(f"Verification complete. Applied {fixes_applied} fixes.")

        # Step 6: Save anonymized CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_csv(output_path, headers, anonymized_rows)
        print(f"Saved anonymized CSV to: {output_path}")

        # Step 6: Save JSON with details (if debug mode)
        if self.config.save_debug_files:
            json_output_path = output_path.with_suffix('.json')
            self._save_json_output(
                dates_shifted,
                pii_redactions,
                input_path,
                output_path,
                json_output_path
            )
            print(f"Saved anonymization details to: {json_output_path}")

    def _read_csv(self, file_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
        """Read CSV file."""
        rows = []
        headers = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = [h for h in (reader.fieldnames or []) if h is not None]
            rows = [{k: v for k, v in row.items() if k is not None} for row in reader]

        return rows, headers

    def _write_csv(self, file_path: Path, headers: List[str], rows: List[Dict[str, str]]) -> None:
        """Write CSV file."""
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def _format_csv_for_llm(self, rows: List[Dict[str, str]], headers: List[str], start_idx: int = 0) -> str:
        """Format CSV data for LLM prompt."""
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

    def _extract_dates_with_regex(self, text: str) -> List[str]:
        """
        Extract all date/time patterns from text using regex.
        
        Returns a list of unique date strings found.
        """
        patterns = [
            # ISO format with time: 2140-09-25 07:15:00 or 2140-09-25T07:15:00
            r'\b(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})\b',
            # Date with AM/PM time: 2140-09-25 07:15PM or 2140-09-25 07:15 PM
            r'\b(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}\s*[AaPp][Mm])\b',
            # ISO date only: 2140-09-25
            r'\b(\d{4}-\d{2}-\d{2})\b',
            # European format: 25.09.2140
            r'\b(\d{2}\.\d{2}\.\d{4})\b',
            # US format with slashes: 09/25/2140
            r'\b(\d{2}/\d{2}/\d{4})\b',
            # Month name formats: September 25, 2140 or Sep 25, 2140
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b',
            # Date with month name: 25 September 2140
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b',
        ]
        
        found_dates: Set[str] = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_dates.update(matches)
        
        # Sort by length descending to process longer (more specific) dates first
        # This helps avoid partial replacements (e.g., replacing "2140-09-25" 
        # before "2140-09-25 07:15PM" could cause issues)
        return sorted(list(found_dates), key=len, reverse=True)

    def _extract_all_dates_from_csv(
        self, 
        rows: List[Dict[str, str]], 
        headers: List[str]
    ) -> List[Tuple[int, str, str]]:
        """
        Extract all dates from CSV using regex.
        
        Returns list of tuples: (row_index, column_name, date_string)
        """
        all_dates: List[Tuple[int, str, str]] = []
        seen: Set[Tuple[int, str, str]] = set()
        
        for row_idx, row in enumerate(rows):
            for col_name in headers:
                value = str(row.get(col_name, ""))
                if not value:
                    continue
                    
                dates = self._extract_dates_with_regex(value)
                for date_str in dates:
                    key = (row_idx, col_name, date_str)
                    if key not in seen:
                        seen.add(key)
                        all_dates.append(key)
        
        return all_dates

    def _phase1_shift_times(
        self,
        rows: List[Dict[str, str]],
        headers: List[str]
    ) -> tuple[List[Dict[str, str]], List[DateTimeShift]]:
        """
        Phase 1: Find and shift all dates/times in CSV using regex extraction.

        Uses regex patterns to reliably extract all dates, then shifts them
        using the shift_datetime tool.

        Args:
            rows: CSV rows as list of dicts
            headers: Column headers

        Returns:
            Tuple of (rows with shifted dates, list of shifts made)
        """
        # Work on a copy
        modified_rows = [row.copy() for row in rows]
        all_shifts: List[DateTimeShift] = []
        
        # Extract all dates using regex
        print("  Extracting dates using pattern matching...")
        all_dates = self._extract_all_dates_from_csv(rows, headers)
        print(f"  Found {len(all_dates)} date occurrences to process")
        
        if not all_dates:
            return modified_rows, all_shifts
        
        # Track unique date strings and their shifted values
        # to avoid calling the tool multiple times for the same date
        shifted_cache: Dict[str, str] = {}
        
        # Process dates and track shifts
        for row_idx, col_name, date_str in all_dates:
            # Check cache first
            if date_str in shifted_cache:
                shifted_value = shifted_cache[date_str]
            else:
                # Call shift_datetime tool
                try:
                    result = shift_datetime.invoke({
                        "datetime_str": date_str,
                        "offset_days": self.time_offset_days
                    })
                    
                    if "[SHIFT_FAILED]" in result:
                        print(f"    Skip (invalid): {date_str}")
                        shifted_cache[date_str] = date_str  # Keep original
                        continue
                    
                    shifted_value = result
                    shifted_cache[date_str] = shifted_value
                    print(f"    Shifted: {date_str} → {shifted_value}")
                    
                except Exception as e:
                    print(f"    Error shifting {date_str}: {e}")
                    shifted_cache[date_str] = date_str  # Keep original
                    continue
            
            # Apply shift if value changed
            if shifted_value != date_str:
                current_value = str(modified_rows[row_idx].get(col_name, ""))
                if date_str in current_value:
                    new_value = current_value.replace(date_str, shifted_value)
                    modified_rows[row_idx][col_name] = new_value
                    
                    all_shifts.append(DateTimeShift(
                        row_index=row_idx,
                        column_name=col_name,
                        original_value=date_str,
                        shifted_value=shifted_value
                    ))
        
        # Summary by row
        rows_with_shifts = len(set(s.row_index for s in all_shifts))
        print(f"  Processed {rows_with_shifts} row(s) with date shifts")

        return modified_rows, all_shifts

    def _phase2_anonymize_pii(
        self,
        rows: List[Dict[str, str]],
        headers: List[str]
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Phase 2: Anonymize all PII using agentic tool-calling with redact_text.

        The LLM identifies PII and calls the redact_text tool for each item found.
        Redactions are applied in real-time as the tools are called.

        Args:
            rows: CSV rows (with dates already shifted)
            headers: Column headers

        Returns:
            Tuple of (anonymized rows, number of redactions applied)
        """
        modified_rows = [row.copy() for row in rows]
        total_redactions = 0

        # Process in batches for large files
        batch_size = 30
        total_batches = (len(rows) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(rows))
            batch_rows = modified_rows[start_idx:end_idx]

            print(f"  Phase 2 - Batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx-1})")

            csv_preview = self._format_csv_for_llm(batch_rows, headers, start_idx)

            prompt = f"""You are a PII anonymization agent. Analyze this CSV data and redact all Personal Identifiable Information (PII).

IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

CSV Data (rows {start_idx} to {end_idx-1}):
{csv_preview}

You have the redact_text tool available. For EACH piece of PII you find, call:
  redact_text(text_to_redact="exact PII text", row_index=N, column_name="column")

PII categories to redact:
- Patient names, physician names, doctor names, staff names
- Physical addresses, specific facility/hospital names
- Patient IDs, medical record numbers, unit numbers
- Phone numbers, fax numbers
- Email addresses
- Ages (specific numbers like "62 years old")

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

Example: If you see "Name: John Smith" in row 0, column "text":
  → call redact_text(text_to_redact="John Smith", row_index=0, column_name="text")
"""

            messages = [HumanMessage(content=prompt)]
            batch_redactions = 0

            # Agentic loop
            max_iterations = 50  # Allow many iterations for thorough redaction
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                try:
                    response = self.llm_anonymize.invoke(messages)
                    messages.append(response)

                    if not response.tool_calls:
                        # No more tool calls - agent is done with this batch
                        break

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        if tool_name == "redact_text":
                            text_to_redact = tool_args.get("text_to_redact", "")
                            row_idx = tool_args.get("row_index", 0)
                            col_name = tool_args.get("column_name", "")

                            # Get the redacted version (asterisks)
                            result = redact_text.invoke(tool_args)

                            if "[REDACT_FAILED" not in result and text_to_redact:
                                # Apply to the specific cell
                                if 0 <= row_idx < len(modified_rows):
                                    cell_value = str(modified_rows[row_idx].get(col_name, ""))
                                    if text_to_redact in cell_value:
                                        modified_rows[row_idx][col_name] = cell_value.replace(
                                            text_to_redact, result
                                        )
                                        batch_redactions += 1
                                        # Truncate long text for display
                                        display_text = text_to_redact[:30] + "..." if len(text_to_redact) > 30 else text_to_redact
                                        print(f"    Redacted: '{display_text}' in row {row_idx}, col '{col_name}'")

                            messages.append(ToolMessage(
                                content=f"Redacted: '{text_to_redact}' → '{result}'",
                                tool_call_id=tool_call["id"]
                            ))

                except Exception as e:
                    print(f"    Error in batch {batch_num + 1}: {e}")
                    break

            total_redactions += batch_redactions

        return modified_rows, total_redactions

    def _phase3_verify_and_fix(
        self,
        original_rows: List[Dict[str, str]],
        anonymized_rows: List[Dict[str, str]],
        headers: List[str]
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Phase 3: Verification agent checks the anonymized output and fixes any issues.

        The agent compares original and anonymized data to identify:
        1. Unshifted dates (dates that appear in both original and anonymized)
        2. Unredacted PII (names, IDs, etc. that weren't anonymized)
        3. Over-redaction (non-PII that was incorrectly redacted)

        Args:
            original_rows: Original CSV rows (before anonymization)
            anonymized_rows: Anonymized CSV rows
            headers: Column headers

        Returns:
            Tuple of (fixed rows, number of fixes applied)
        """
        modified_rows = [row.copy() for row in anonymized_rows]
        total_fixes = 0

        # Process in batches for large files
        batch_size = 10
        total_batches = (len(original_rows) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(original_rows))

            print(f"  Verifying batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx - 1})")

            # Prepare comparison data for this batch
            comparison_data = self._format_comparison_for_llm(
                original_rows[start_idx:end_idx],
                modified_rows[start_idx:end_idx],
                headers,
                start_idx
            )

            prompt = f"""You are a verification agent for medical data anonymization.
Compare the ORIGINAL and ANONYMIZED data below and identify any issues.

{comparison_data}

Time offset used: {self.time_offset_days} days

You have THREE tools available:
1. shift_datetime - to fix unshifted dates
2. redact_text - to redact PII that was missed
3. restore_text - to fix over-redaction (restore incorrectly redacted content)

Your tasks:

1. CHECK FOR UNSHIFTED DATES: Look for dates in the ANONYMIZED data that are identical to the ORIGINAL.
   All dates should be shifted by {self.time_offset_days} days.
   → Use shift_datetime(datetime_str, offset_days={self.time_offset_days}) to fix

2. CHECK FOR UNREDACTED PII: Look for PII that appears UNCHANGED in the anonymized version:
   - Patient names, doctor names, staff names (e.g., "Laura Martinez", "Dr. Smith")
   - Phone numbers (e.g., "555-123-4567")
   - Fax numbers, email addresses
   - Specific addresses or facility names that identify location
   → Use redact_text(text_to_redact, row_index, column_name) to fix

3. CHECK FOR OVER-REDACTION: Look for content that was INCORRECTLY redacted (asterisks where there shouldn't be):
   - Medical terminology (diabetes, hypertension, pneumonia, etc.)
   - Medication names (metformin, lisinopril, aspirin, etc.)
   - Procedure names (colonoscopy, MRI, CT scan, etc.)
   - Generic locations (EMERGENCY ROOM, ICU, HOME, etc.)
   - Dates that were redacted instead of shifted
   → Use restore_text(redacted_text, original_text, row_index, column_name) to fix
   
   Example: If "diabetes" was redacted to "********", call:
   restore_text(redacted_text="********", original_text="diabetes", row_index=0, column_name="text")

IMPORTANT:
- Compare ORIGINAL vs ANONYMIZED to identify issues
- Only redact actual PII, not medical content
- Restore any medical terms that were incorrectly redacted

Call the appropriate tool for each issue you find. When done, summarize your findings.
"""

            messages = [HumanMessage(content=prompt)]

            # Agentic loop for verification
            max_iterations = 30
            iteration = 0
            batch_fixes = 0

            while iteration < max_iterations:
                iteration += 1

                try:
                    response = self.llm_verify.invoke(messages)
                    messages.append(response)

                    if not response.tool_calls:
                        # No more tool calls - agent is done
                        break

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        if tool_name == "shift_datetime":
                            original_date = tool_args.get("datetime_str", "")
                            result = shift_datetime.invoke(tool_args)

                            if "[SHIFT_FAILED]" not in result:
                                # Apply the fix to all occurrences in this batch
                                for row_idx in range(start_idx, end_idx):
                                    for col_name in headers:
                                        cell_value = str(modified_rows[row_idx].get(col_name, ""))
                                        if original_date in cell_value:
                                            modified_rows[row_idx][col_name] = cell_value.replace(
                                                original_date, result
                                            )
                                            batch_fixes += 1
                                            print(f"    Fixed date: {original_date} → {result}")

                            messages.append(ToolMessage(
                                content=f"Date shifted: {original_date} → {result}",
                                tool_call_id=tool_call["id"]
                            ))

                        elif tool_name == "redact_text":
                            text_to_redact = tool_args.get("text_to_redact", "")
                            row_idx = tool_args.get("row_index", 0)
                            col_name = tool_args.get("column_name", "")
                            
                            # Get the redacted version (asterisks)
                            result = redact_text.invoke(tool_args)
                            
                            if "[REDACT_FAILED" not in result:
                                # Apply to the specific cell
                                if start_idx <= row_idx < end_idx:
                                    cell_value = str(modified_rows[row_idx].get(col_name, ""))
                                    if text_to_redact in cell_value:
                                        modified_rows[row_idx][col_name] = cell_value.replace(
                                            text_to_redact, result
                                        )
                                        batch_fixes += 1
                                        print(f"    Fixed PII: '{text_to_redact}' → '{result}'")

                            messages.append(ToolMessage(
                                content=f"Text redacted: '{text_to_redact}' → '{result}'",
                                tool_call_id=tool_call["id"]
                            ))

                        elif tool_name == "restore_text":
                            redacted_text = tool_args.get("redacted_text", "")
                            original_text = tool_args.get("original_text", "")
                            row_idx = tool_args.get("row_index", 0)
                            col_name = tool_args.get("column_name", "")
                            
                            # Get the original text to restore
                            result = restore_text.invoke(tool_args)
                            
                            if "[RESTORE_FAILED" not in result:
                                # Apply to the specific cell - replace redacted with original
                                if start_idx <= row_idx < end_idx:
                                    cell_value = str(modified_rows[row_idx].get(col_name, ""))
                                    if redacted_text in cell_value:
                                        modified_rows[row_idx][col_name] = cell_value.replace(
                                            redacted_text, result
                                        )
                                        batch_fixes += 1
                                        print(f"    Restored: '{redacted_text}' → '{result}'")

                            messages.append(ToolMessage(
                                content=f"Text restored: '{redacted_text}' → '{result}'",
                                tool_call_id=tool_call["id"]
                            ))

                except Exception as e:
                    print(f"    Verification error: {e}")
                    break

            total_fixes += batch_fixes

        return modified_rows, total_fixes

    def _format_comparison_for_llm(
        self,
        original_rows: List[Dict[str, str]],
        anonymized_rows: List[Dict[str, str]],
        headers: List[str],
        start_idx: int = 0
    ) -> str:
        """Format original and anonymized data side-by-side for comparison."""
        lines = []
        lines.append("=" * 60)
        lines.append("COMPARISON: ORIGINAL vs ANONYMIZED")
        lines.append("=" * 60)

        for idx, (orig, anon) in enumerate(zip(original_rows, anonymized_rows)):
            absolute_idx = start_idx + idx
            lines.append(f"\n=== ROW {absolute_idx} ===")

            for header in headers:
                orig_val = orig.get(header, "")
                anon_val = anon.get(header, "")

                # Only show if there's content and it changed
                if orig_val or anon_val:
                    # Truncate very long values for the prompt
                    orig_display = orig_val[:500] + "..." if len(orig_val) > 500 else orig_val
                    anon_display = anon_val[:500] + "..." if len(anon_val) > 500 else anon_val

                    if orig_val != anon_val:
                        lines.append(f"\n[{header}] (CHANGED)")
                        lines.append(f"  ORIGINAL: {orig_display}")
                        lines.append(f"  ANONYMIZED: {anon_display}")
                    else:
                        lines.append(f"\n[{header}] (unchanged): {orig_display[:200]}...")

        return "\n".join(lines)

    def _save_json_output(
        self,
        dates_shifted: List[DateTimeShift],
        pii_redactions_count: int,
        input_path: Path,
        output_path: Path,
        json_output_path: Path
    ) -> None:
        """Save anonymization details as JSON."""
        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "agentic_csv_anonymization",
                "time_offset_days": self.time_offset_days,
                "total_dates_shifted": len(dates_shifted),
                "total_pii_redactions": pii_redactions_count
            },
            "phase1_time_shifts": [
                {
                    "row_index": d.row_index,
                    "column_name": d.column_name,
                    "original": d.original_value,
                    "shifted": d.shifted_value
                }
                for d in dates_shifted
            ]
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
