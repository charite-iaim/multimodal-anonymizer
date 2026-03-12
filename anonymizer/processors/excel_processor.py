"""
Agentic Excel file processor using LLM with tool-calling for anonymization.

This processor uses a two-step agentic approach (same as CSV processor):
1. LLM identifies and shifts dates/times using the shift_datetime tool
2. LLM anonymizes all other PII (but not the already-shifted dates)

Supports .xlsx and .xls files with multiple sheets.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..llm_factory import create_chat_llm
from ..tools.time_shift_tool import shift_datetime
from ..tools.redact_tool import redact_text, redact_column
from ..retry_utils import retry_with_backoff, RetryConfig, create_retry_callback
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG


class DateTimeShift(BaseModel):
    """A date/time shift in the Excel file."""
    sheet_name: str = Field(description="Sheet name")
    row_index: int = Field(description="Row index (0-based)")
    column_name: str = Field(description="Column name")
    original_value: str = Field(description="Original date/time value")
    shifted_value: str = Field(description="Shifted date/time value")


class ExcelProcessor(FileProcessor):
    """
    Processor for Excel files using LLM with tool-calling.

    This processor implements a two-phase approach:
    1. Time-Shift Phase: LLM uses the shift_datetime tool to find and shift all dates
    2. Anonymization Phase: LLM anonymizes all other PII (names, addresses, IDs, etc.)

    Supports multiple sheets - each sheet is processed independently.
    """

    def __init__(
        self,
        config: AnonymizerConfig,
        time_offset_days: Optional[int] = None,
        max_workers: int = 4,
        batch_size_phase2: int = 15,
        batch_size_phase3: int = 15,
        prompt_config: Optional[PromptConfig] = None
    ):
        """
        Initialize agentic Excel processor.

        Args:
            config: Configuration object with LLM settings
            time_offset_days: Fixed offset for time shifting. If None, a random offset is generated.
            max_workers: Maximum number of parallel workers for batch processing.
            batch_size_phase2: Number of rows per batch in Phase 2 (PII anonymization).
            batch_size_phase3: Number of rows per batch in Phase 3 (verification).
            prompt_config: Custom prompt configuration. If None, uses default prompts.
        """
        super().__init__(config)

        # Prompt configuration
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG

        # Parallelization settings
        self.max_workers = max_workers
        self.batch_size_phase2 = batch_size_phase2
        self.batch_size_phase3 = batch_size_phase3

        # Thread-safe lock for print statements
        self._print_lock = threading.Lock()


        # Generate random offset if not provided
        if time_offset_days is None:
            # Random offset between 1-3 years in days
            self.time_offset_days = random.randint(365, 1095)

            # Random sign (+/-)
            if random.random() < 0.5:
                self.time_offset_days = -self.time_offset_days
        else:
            self.time_offset_days = time_offset_days

        # Configure retry settings for LLM calls
        self.retry_config = RetryConfig(
            max_retries=3,
            initial_delay=2.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Store config for creating LLM instances in worker threads
        self._config = config

        # Initialize LLM with tools for phase 1 (time shifting)
        self.llm_with_tools = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[shift_datetime],
        )

        # Initialize LLM with tools for phase 2 (PII anonymization) - main thread fallback
        self.llm_anonymize = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[redact_text, redact_column],
        )

        # Initialize LLM with tools for phase 3 (verification) - main thread fallback
        self.llm_verify = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[shift_datetime, redact_text],
        )

    def _create_llm_anonymize(self):
        """Create a new LLM instance for anonymization (thread-safe)."""
        return create_chat_llm(
            config=self._config,
            timeout=600,
            max_tokens=16000,
            tools=[redact_text, redact_column],
        )

    def _create_llm_verify(self):
        """Create a new LLM instance for verification (thread-safe)."""
        return create_chat_llm(
            config=self._config,
            timeout=600,
            max_tokens=16000,
            tools=[shift_datetime, redact_text],
        )

    def _safe_print(self, message: str) -> None:
        """Thread-safe print."""
        with self._print_lock:
            print(message)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is an Excel file."""
        return file_path.suffix.lower() in [".xlsx", ".xls"]

    def extract_content(self, file_path: Path) -> str:
        """Extract Excel content as string representation."""
        excel_file = pd.ExcelFile(file_path)
        content_parts = []

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            content_parts.append(f"=== Sheet: {sheet_name} ===\n{df.to_string()}")

        return "\n\n".join(content_parts)

    def anonymize(self, input_path: Path, output_path: Path, verify: bool = True) -> None:
        """
        Anonymize Excel file using agentic LLM approach.

        Steps:
        1. Read Excel file (all sheets)
        2. Phase 1: Extract and shift all dates/times using regex
        3. Phase 2: Use LLM to anonymize all other PII
        4. Phase 3: Verification agent checks and fixes any issues
        5. Save anonymized Excel

        Args:
            input_path: Path to input Excel file
            output_path: Path to save anonymized Excel file
            verify: Whether to run the verification phase (default: True)
        """
        # Convert to Path if string
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        output_path = Path(output_path) if isinstance(output_path, str) else output_path

        print(f"Processing Excel: {input_path.name}")
        print(f"Time offset: {self.time_offset_days} days")

        # Step 1: Read Excel file (all sheets)
        sheets_data = self._read_excel(input_path)
        print(f"Found {len(sheets_data)} sheet(s): {', '.join(sheets_data.keys())}")

        if not sheets_data:
            print("Empty Excel file, saving as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(input_path, output_path)
            return

        # Process each sheet
        all_dates_shifted: List[DateTimeShift] = []
        total_pii_redactions = 0
        anonymized_sheets: Dict[str, Tuple[List[Dict[str, str]], List[str]]] = {}

        for sheet_name, (rows, headers) in sheets_data.items():
            print(f"\n{'='*60}")
            print(f"Processing sheet: {sheet_name}")
            print(f"Found {len(rows)} rows with columns: {', '.join(headers)}")
            print(f"{'='*60}")

            if not rows:
                print(f"Empty sheet '{sheet_name}', keeping as-is")
                anonymized_sheets[sheet_name] = (rows, headers)
                continue

            original_rows = [row.copy() for row in rows]

            # Step 2: Phase 1 - Time shifting with regex extraction
            print(f"\n=== Phase 1: Time Shifting ({sheet_name}) ===")
            shifted_rows, dates_shifted = self._phase1_shift_times(rows, headers, sheet_name)
            all_dates_shifted.extend(dates_shifted)
            print(f"Shifted {len(dates_shifted)} date/time values")

            # Step 3a: Phase 2a - Identify and redact entire PII columns
            print(f"\n=== Phase 2a: Column-Level PII Detection ({sheet_name}) ===")
            column_redacted_rows, columns_redacted = self._phase2a_identify_pii_columns(shifted_rows, headers)
            print(f"Redacted {len(columns_redacted)} entire column(s): {', '.join(columns_redacted) if columns_redacted else 'none'}")

            # Step 3b: Phase 2b - Anonymize other PII (agentic with redact_text tool)
            print(f"\n=== Phase 2b: PII Anonymization ({sheet_name}) ===")
            anonymized_rows, pii_redactions = self._phase2b_anonymize_pii(column_redacted_rows, headers, columns_redacted)
            total_pii_redactions += pii_redactions
            print(f"Applied {pii_redactions} PII redactions")

            # Step 4: Phase 3 - Verification (optional but recommended)
            if verify:
                print(f"\n=== Phase 3: Iterative Verification ({sheet_name}) ===")
                max_iterations = 3
                total_fixes = 0

                for iteration in range(max_iterations):
                    print(f"\n  Verification pass {iteration + 1}/{max_iterations}...")
                    anonymized_rows, fixes_applied = self._phase3_verify_and_fix(
                        original_rows, anonymized_rows, headers
                    )
                    total_fixes += fixes_applied
                    print(f"  Pass {iteration + 1}: Applied {fixes_applied} fixes.")

                    if fixes_applied == 0:
                        print(f"  No more issues found. Verification complete after {iteration + 1} pass(es).")
                        break
                else:
                    print(f"  Warning: Completed {max_iterations} passes with {total_fixes} total fixes.")
                    print(f"  Consider reviewing the output manually for any remaining PIIs.")

                print(f"\nVerification summary for '{sheet_name}': Applied {total_fixes} total fixes.")

            anonymized_sheets[sheet_name] = (anonymized_rows, headers)

        # Step 5: Save anonymized Excel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_excel(output_path, anonymized_sheets)
        print(f"\nSaved anonymized Excel to: {output_path}")

        # Step 6: Save JSON with details (if debug mode)
        if self.config.save_debug_files:
            json_output_path = output_path.with_suffix('.json')
            self._save_json_output(
                all_dates_shifted,
                total_pii_redactions,
                input_path,
                output_path,
                json_output_path
            )
            print(f"Saved anonymization details to: {json_output_path}")

    def _read_excel(self, file_path: Path) -> Dict[str, Tuple[List[Dict[str, str]], List[str]]]:
        """
        Read Excel file with all sheets.

        Returns:
            Dict mapping sheet_name -> (rows as list of dicts, headers as list)
        """
        sheets_data = {}

        excel_file = pd.ExcelFile(file_path)

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Convert all values to strings to handle mixed types
            df = df.fillna("")
            df = df.astype(str)

            # Replace "nan" strings that come from NaN values
            df = df.replace("nan", "")

            headers = list(df.columns)
            rows = df.to_dict(orient='records')

            sheets_data[sheet_name] = (rows, headers)

        return sheets_data

    def _write_excel(self, file_path: Path, sheets_data: Dict[str, Tuple[List[Dict[str, str]], List[str]]]) -> None:
        """
        Write Excel file with multiple sheets.

        Args:
            file_path: Output path
            sheets_data: Dict mapping sheet_name -> (rows, headers)
        """
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, (rows, headers) in sheets_data.items():
                if rows:
                    df = pd.DataFrame(rows, columns=headers)
                else:
                    df = pd.DataFrame(columns=headers)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _format_csv_for_llm(self, rows: List[Dict[str, str]], headers: List[str], start_idx: int = 0) -> str:
        """Format data for LLM prompt (same format as CSV)."""
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
        """Extract all date/time patterns from text using regex."""
        patterns = [
            r'\b(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})\b',
            r'\b(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}\s*[AaPp][Mm])\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{2}\.\d{2}\.\d{4})\b',
            r'\b(\d{2}/\d{2}/\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b',
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b',
        ]

        found_dates: Set[str] = set()

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_dates.update(matches)

        return sorted(list(found_dates), key=len, reverse=True)

    def _extract_all_dates_from_rows(
        self,
        rows: List[Dict[str, str]],
        headers: List[str]
    ) -> List[Tuple[int, str, str]]:
        """Extract all dates from rows using regex."""
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
        headers: List[str],
        sheet_name: str = "Sheet1"
    ) -> Tuple[List[Dict[str, str]], List[DateTimeShift]]:
        """Phase 1: Find and shift all dates/times using regex extraction."""
        modified_rows = [row.copy() for row in rows]
        all_shifts: List[DateTimeShift] = []

        print("  Extracting dates using pattern matching...")
        all_dates = self._extract_all_dates_from_rows(rows, headers)
        print(f"  Found {len(all_dates)} date occurrences to process")

        if not all_dates:
            return modified_rows, all_shifts

        shifted_cache: Dict[str, str] = {}

        for row_idx, col_name, date_str in all_dates:
            if date_str in shifted_cache:
                shifted_value = shifted_cache[date_str]
            else:
                try:
                    result = shift_datetime.invoke({
                        "datetime_str": date_str,
                        "offset_days": self.time_offset_days
                    })

                    if "[SHIFT_FAILED]" in result:
                        print(f"    Skip (invalid): {date_str}")
                        shifted_cache[date_str] = date_str
                        continue

                    shifted_value = result
                    shifted_cache[date_str] = shifted_value
                    print(f"    Shifted: {date_str} → {shifted_value}")

                except Exception as e:
                    print(f"    Error shifting {date_str}: {e}")
                    shifted_cache[date_str] = date_str
                    continue

            if shifted_value != date_str:
                current_value = str(modified_rows[row_idx].get(col_name, ""))
                if date_str in current_value:
                    new_value = current_value.replace(date_str, shifted_value)
                    modified_rows[row_idx][col_name] = new_value

                    all_shifts.append(DateTimeShift(
                        sheet_name=sheet_name,
                        row_index=row_idx,
                        column_name=col_name,
                        original_value=date_str,
                        shifted_value=shifted_value
                    ))

        rows_with_shifts = len(set(s.row_index for s in all_shifts))
        print(f"  Processed {rows_with_shifts} row(s) with date shifts")

        return modified_rows, all_shifts

    def _phase2a_identify_pii_columns(
        self,
        rows: List[Dict[str, str]],
        headers: List[str]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Phase 2a: Use LLM to identify columns that contain PII."""
        if not rows:
            return rows, []

        sample_size = min(5, len(rows))
        sample_rows = rows[:sample_size]
        sample_preview = self._format_csv_for_llm(sample_rows, headers, 0)

        prompt = self.prompt_config.get_column_detection_prompt(
            columns=', '.join(headers),
            sample_data=sample_preview
        )

        messages = [HumanMessage(content=prompt)]
        columns_to_redact: List[str] = []

        retry_callback = create_retry_callback(
            lambda msg: self._safe_print(f"    {msg}")
        )

        def invoke_with_retry(msgs):
            return retry_with_backoff(
                lambda: self.llm_anonymize.invoke(msgs),
                config=self.retry_config,
                on_retry=retry_callback,
            )

        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                response = invoke_with_retry(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "redact_column":
                        col_name = tool_args.get("column_name", "")
                        reason = tool_args.get("reason", "")

                        result = redact_column.invoke(tool_args)

                        if "[REDACT_COLUMN_FAILED" not in result and col_name in headers:
                            if col_name not in columns_to_redact:
                                columns_to_redact.append(col_name)
                                self._safe_print(f"    Column '{col_name}' marked for redaction: {reason}")

                        messages.append(ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        ))
                    elif tool_name == "redact_text":
                        messages.append(ToolMessage(
                            content="[Ignored in column detection phase - use redact_column instead]",
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                self._safe_print(f"    Error in column detection: {e}")
                break

        modified_rows = [row.copy() for row in rows]

        if columns_to_redact:
            for row in modified_rows:
                for col_name in columns_to_redact:
                    if col_name in row and row[col_name]:
                        original_value = row[col_name]
                        row[col_name] = "*" * len(original_value)

        return modified_rows, columns_to_redact

    def _process_single_batch_phase2(
        self,
        batch_rows: List[Dict[str, str]],
        headers: List[str],
        start_idx: int,
        batch_num: int,
        total_batches: int,
        already_redacted_columns: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, str]], int, List[Tuple[int, str, str, str]]]:
        """Process a single batch for Phase 2 (PII anonymization)."""
        if already_redacted_columns is None:
            already_redacted_columns = []

        llm = self._create_llm_anonymize()

        modified_batch = [row.copy() for row in batch_rows]
        end_idx = start_idx + len(batch_rows)
        redactions_applied: List[Tuple[int, str, str, str]] = []

        self._safe_print(f"  Phase 2 - Batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx-1})")

        active_headers = [h for h in headers if h not in already_redacted_columns]
        csv_preview = self._format_csv_for_llm(batch_rows, active_headers, start_idx)

        skipped_cols_str = ', '.join(already_redacted_columns) if already_redacted_columns else ""

        prompt = self.prompt_config.get_csv_anonymization_prompt(
            csv_data=f"Data (rows {start_idx} to {end_idx-1}):\n{csv_preview}",
            skipped_columns=skipped_cols_str
        )

        messages = [HumanMessage(content=prompt)]
        batch_redactions = 0

        retry_callback = lambda attempt, error, delay: self._safe_print(
            f"    [LLM] Retry {attempt}: {type(error).__name__} - waiting {delay:.1f}s"
        )

        def invoke_llm_with_retry(msgs):
            return retry_with_backoff(
                lambda: llm.invoke(msgs),
                config=RetryConfig(
                    max_retries=3,
                    initial_delay=2.0,
                    max_delay=60.0,
                    exponential_base=2.0,
                    jitter=True,
                ),
                on_retry=retry_callback,
            )

        max_iterations = 50
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                response = invoke_llm_with_retry(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "redact_text":
                        text_to_redact = tool_args.get("text_to_redact", "")
                        row_idx = tool_args.get("row_index", 0)
                        col_name = tool_args.get("column_name", "")

                        result = redact_text.invoke(tool_args)

                        if "[REDACT_FAILED" not in result and text_to_redact:
                            local_idx = row_idx - start_idx
                            if 0 <= local_idx < len(modified_batch):
                                cell_value = str(modified_batch[local_idx].get(col_name, ""))
                                if text_to_redact in cell_value:
                                    modified_batch[local_idx][col_name] = cell_value.replace(
                                        text_to_redact, result
                                    )
                                    batch_redactions += 1
                                    redactions_applied.append((row_idx, col_name, text_to_redact, result))
                                    display_text = text_to_redact[:30] + "..." if len(text_to_redact) > 30 else text_to_redact
                                    self._safe_print(f"    Redacted: '{display_text}' in row {row_idx}, col '{col_name}'")

                        messages.append(ToolMessage(
                            content=f"Redacted: '{text_to_redact}' → '{result}'",
                            tool_call_id=tool_call["id"]
                        ))

                    elif tool_name == "redact_column":
                        col_name = tool_args.get("column_name", "")
                        reason = tool_args.get("reason", "")

                        result = redact_column.invoke(tool_args)

                        if "[REDACT_COLUMN_FAILED" not in result and col_name in headers:
                            col_redactions = 0
                            for local_idx in range(len(modified_batch)):
                                cell_value = str(modified_batch[local_idx].get(col_name, ""))
                                if cell_value and not all(c == '*' for c in cell_value):
                                    original_value = cell_value
                                    modified_batch[local_idx][col_name] = "*" * len(cell_value)
                                    col_redactions += 1
                                    redactions_applied.append((start_idx + local_idx, col_name, original_value, "*" * len(original_value)))

                            batch_redactions += col_redactions
                            self._safe_print(f"    Redacted entire column '{col_name}' ({col_redactions} values): {reason}")

                        messages.append(ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                self._safe_print(f"    Error in batch {batch_num + 1}: {e}")
                break

        return modified_batch, batch_redactions, redactions_applied

    def _phase2b_anonymize_pii(
        self,
        rows: List[Dict[str, str]],
        headers: List[str],
        already_redacted_columns: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, str]], int]:
        """Phase 2b: Anonymize all PII using agentic tool-calling."""
        if already_redacted_columns is None:
            already_redacted_columns = []

        batch_size = self.batch_size_phase2
        total_batches = (len(rows) + batch_size - 1) // batch_size

        batches = []
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(rows))
            batch_rows = [rows[i].copy() for i in range(start_idx, end_idx)]
            batches.append((batch_rows, headers, start_idx, batch_num, total_batches, already_redacted_columns))

        results: Dict[int, Tuple[List[Dict[str, str]], int]] = {}
        total_redactions = 0

        if already_redacted_columns:
            print(f"  Skipping already redacted columns: {', '.join(already_redacted_columns)}")
        print(f"  Processing {total_batches} batches with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self._process_single_batch_phase2,
                    batch_rows, hdrs, start_idx, batch_num, total, redacted_cols
                ): start_idx
                for batch_rows, hdrs, start_idx, batch_num, total, redacted_cols in batches
            }

            for future in as_completed(future_to_batch):
                start_idx = future_to_batch[future]
                try:
                    modified_batch, redaction_count, _ = future.result()
                    results[start_idx] = (modified_batch, redaction_count)
                    total_redactions += redaction_count
                except Exception as e:
                    self._safe_print(f"    Batch starting at {start_idx} failed: {e}")
                    batch_num = start_idx // batch_size
                    end_idx = min(start_idx + batch_size, len(rows))
                    results[start_idx] = ([rows[i].copy() for i in range(start_idx, end_idx)], 0)

        modified_rows: List[Dict[str, str]] = []
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            batch_result, _ = results[start_idx]
            modified_rows.extend(batch_result)

        return modified_rows, total_redactions

    def _process_single_batch_phase3(
        self,
        original_batch: List[Dict[str, str]],
        anonymized_batch: List[Dict[str, str]],
        headers: List[str],
        start_idx: int,
        batch_num: int,
        total_batches: int
    ) -> Tuple[List[Dict[str, str]], int]:
        """Process a single batch for Phase 3 (verification)."""
        llm = self._create_llm_verify()

        modified_batch = [row.copy() for row in anonymized_batch]
        end_idx = start_idx + len(anonymized_batch)

        self._safe_print(f"  Verifying batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx - 1})")

        comparison_data = self._format_comparison_for_llm(
            original_batch,
            anonymized_batch,
            headers,
            start_idx
        )

        prompt = self.prompt_config.get_csv_verification_prompt(
            comparison_data=comparison_data,
            time_offset=self.time_offset_days
        )

        messages = [HumanMessage(content=prompt)]

        max_iterations = 30
        iteration = 0
        batch_fixes = 0

        verify_retry_callback = lambda attempt, error, delay: self._safe_print(
            f"    [Verify] Retry {attempt}: {type(error).__name__} - waiting {delay:.1f}s"
        )

        def invoke_verify_with_retry(msgs):
            return retry_with_backoff(
                lambda: llm.invoke(msgs),
                config=RetryConfig(
                    max_retries=3,
                    initial_delay=2.0,
                    max_delay=60.0,
                    exponential_base=2.0,
                    jitter=True,
                ),
                on_retry=verify_retry_callback,
            )

        while iteration < max_iterations:
            iteration += 1

            try:
                response = invoke_verify_with_retry(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "shift_datetime":
                        original_date = tool_args.get("datetime_str", "")
                        result = shift_datetime.invoke(tool_args)

                        if "[SHIFT_FAILED]" not in result:
                            for local_idx in range(len(modified_batch)):
                                for col_name in headers:
                                    cell_value = str(modified_batch[local_idx].get(col_name, ""))
                                    if original_date in cell_value:
                                        modified_batch[local_idx][col_name] = cell_value.replace(
                                            original_date, result
                                        )
                                        batch_fixes += 1
                                        self._safe_print(f"    Fixed date: {original_date} → {result}")

                        messages.append(ToolMessage(
                            content=f"Date shifted: {original_date} → {result}",
                            tool_call_id=tool_call["id"]
                        ))

                    elif tool_name == "redact_text":
                        text_to_redact = tool_args.get("text_to_redact", "")
                        row_idx = tool_args.get("row_index", 0)
                        col_name = tool_args.get("column_name", "")

                        result = redact_text.invoke(tool_args)

                        if "[REDACT_FAILED" not in result:
                            local_idx = row_idx - start_idx
                            if 0 <= local_idx < len(modified_batch):
                                cell_value = str(modified_batch[local_idx].get(col_name, ""))
                                if text_to_redact in cell_value:
                                    modified_batch[local_idx][col_name] = cell_value.replace(
                                        text_to_redact, result
                                    )
                                    batch_fixes += 1
                                    self._safe_print(f"    Fixed PII: '{text_to_redact}' → '{result}'")

                        messages.append(ToolMessage(
                            content=f"Text redacted: '{text_to_redact}' → '{result}'",
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                self._safe_print(f"    Verification error in batch {batch_num + 1}: {e}")
                break

        return modified_batch, batch_fixes

    def _phase3_verify_and_fix(
        self,
        original_rows: List[Dict[str, str]],
        anonymized_rows: List[Dict[str, str]],
        headers: List[str]
    ) -> Tuple[List[Dict[str, str]], int]:
        """Phase 3: Verification agent checks and fixes any issues."""
        batch_size = self.batch_size_phase3
        total_batches = (len(original_rows) + batch_size - 1) // batch_size

        batches = []
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(original_rows))
            original_batch = [original_rows[i].copy() for i in range(start_idx, end_idx)]
            anonymized_batch = [anonymized_rows[i].copy() for i in range(start_idx, end_idx)]
            batches.append((original_batch, anonymized_batch, headers, start_idx, batch_num, total_batches))

        results: Dict[int, Tuple[List[Dict[str, str]], int]] = {}
        total_fixes = 0

        print(f"  Verifying {total_batches} batches with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self._process_single_batch_phase3,
                    orig_batch, anon_batch, hdrs, start_idx, batch_num, total
                ): start_idx
                for orig_batch, anon_batch, hdrs, start_idx, batch_num, total in batches
            }

            for future in as_completed(future_to_batch):
                start_idx = future_to_batch[future]
                try:
                    modified_batch, fix_count = future.result()
                    results[start_idx] = (modified_batch, fix_count)
                    total_fixes += fix_count
                except Exception as e:
                    self._safe_print(f"    Verification batch at {start_idx} failed: {e}")
                    batch_num = start_idx // batch_size
                    end_idx = min(start_idx + batch_size, len(anonymized_rows))
                    results[start_idx] = ([anonymized_rows[i].copy() for i in range(start_idx, end_idx)], 0)

        modified_rows: List[Dict[str, str]] = []
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            batch_result, _ = results[start_idx]
            modified_rows.extend(batch_result)

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

                if orig_val or anon_val:
                    if orig_val != anon_val:
                        lines.append(f"\n[{header}] (CHANGED)")
                        lines.append(f"  ORIGINAL: {orig_val}")
                        lines.append(f"  ANONYMIZED: {anon_val}")
                    else:
                        orig_display = orig_val[:200] + "..." if len(orig_val) > 200 else orig_val
                        lines.append(f"\n[{header}] (unchanged): {orig_display}")

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
                "processing_method": "agentic_excel_anonymization",
                "time_offset_days": self.time_offset_days,
                "total_dates_shifted": len(dates_shifted),
                "total_pii_redactions": pii_redactions_count
            },
            "phase1_time_shifts": [
                {
                    "sheet_name": d.sheet_name,
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
