"""
Agentic text file processor using LLM with tool-calling for anonymization.

This processor uses a three-step agentic approach (same as CSV processor):
1. Phase 1: Regex-based date extraction + shift_datetime tool
2. Phase 2: LLM identifies and redacts PII using redact_text tool
3. Phase 3: Verification agent checks and fixes any issues
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
import random

from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..llm_factory import create_chat_llm
from ..tools.time_shift_tool import shift_datetime, redact_text, restore_text
from ..retry_utils import retry_with_backoff, RetryConfig, create_retry_callback


class DateTimeShift(BaseModel):
    """A date/time shift in the text."""
    original_value: str = Field(description="Original date/time value")
    shifted_value: str = Field(description="Shifted date/time value")
    context: str = Field(description="Brief context where this date was found")


class AgenticTextProcessor(FileProcessor):
    """
    Agentic processor for text files using LLM with tool-calling.

    This processor implements a three-phase approach (matching CSV processor):
    1. Time-Shift Phase: Regex extraction + shift_datetime tool for all dates
    2. Anonymization Phase: LLM uses redact_text tool for PII
    3. Verification Phase: LLM verifies and fixes any issues
    """

    def __init__(self, config: AnonymizerConfig, time_offset_days: Optional[int] = None):
        """
        Initialize agentic text processor.

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

        # Configure retry settings for LLM calls
        self.retry_config = RetryConfig(
            max_retries=3,
            initial_delay=2.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Initialize LLM with tools for phase 1 (time shifting - only needed if LLM finds additional dates)
        self.llm_with_tools = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[shift_datetime],
        )

        # Initialize LLM with tools for phase 2 (PII anonymization)
        self.llm_anonymize = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[redact_text],
        )

        # Initialize LLM with tools for phase 3 (verification)
        self.llm_verify = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[shift_datetime, redact_text, restore_text],
        )

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a text file (.txt or .hea ECG header)."""
        return file_path.suffix.lower() in [".txt", ".hea"]

    def extract_content(self, file_path: Path) -> str:
        """Extract text content as string."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def anonymize(self, input_path: Path, output_path: Path, verify: bool = True) -> None:
        """
        Anonymize text file using agentic LLM approach.

        Steps:
        1. Read text file
        2. Phase 1: Extract and shift all dates/times using regex
        3. Phase 2: Use LLM with redact_text tool to anonymize PII
        4. Phase 3: Verification agent checks and fixes any issues
        5. Save anonymized text

        Args:
            input_path: Path to input text file
            output_path: Path to save anonymized text file
            verify: Whether to run the verification phase (default: True)
        """
        # Convert to Path if string
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        output_path = Path(output_path) if isinstance(output_path, str) else output_path

        print(f"Processing (Agentic): {input_path.name}")
        print(f"Time offset: {self.time_offset_days} days")

        # Step 1: Read text file
        content = self.extract_content(input_path)
        original_content = content  # Keep original for verification
        print(f"Read {len(content)} characters")

        if not content.strip():
            print("Empty text file, saving as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return

        # Step 2: Phase 1 - Time shifting with regex extraction
        print("\n=== Phase 1: Time Shifting ===")
        shifted_content, dates_shifted = self._phase1_shift_times(content)
        print(f"Shifted {len(dates_shifted)} date/time values")

        # Step 3: Phase 2 - Anonymize other PII (agentic with redact_text tool)
        print("\n=== Phase 2: PII Anonymization ===")
        anonymized_content, pii_redactions = self._phase2_anonymize_pii(shifted_content)
        print(f"Applied {pii_redactions} PII redactions")

        # Step 4: Phase 3 - Verification (optional but recommended)
        # Use iterative verification to catch all remaining PIIs
        if verify:
            print("\n=== Phase 3: Iterative Verification ===")
            max_iterations = 3  # Maximum number of verification passes
            total_fixes = 0
            
            for iteration in range(max_iterations):
                print(f"\n  Verification pass {iteration + 1}/{max_iterations}...")
                anonymized_content, fixes_applied = self._phase3_verify_and_fix(
                    original_content, anonymized_content
                )
                total_fixes += fixes_applied
                print(f"  Pass {iteration + 1}: Applied {fixes_applied} fixes.")
                
                # Stop if no more fixes were needed
                if fixes_applied == 0:
                    print(f"  No more issues found. Verification complete after {iteration + 1} pass(es).")
                    break
            else:
                # All iterations completed but still finding issues
                print(f"  Warning: Completed {max_iterations} passes with {total_fixes} total fixes.")
                print(f"  Consider reviewing the output manually for any remaining PIIs.")
            
            print(f"\nVerification summary: Applied {total_fixes} total fixes across all passes.")

        # Step 5: Save anonymized text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(anonymized_content)
        print(f"Saved anonymized text to: {output_path}")

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
        return sorted(list(found_dates), key=len, reverse=True)

    def _phase1_shift_times(self, content: str) -> Tuple[str, List[DateTimeShift]]:
        """
        Phase 1: Find and shift all dates/times using regex extraction.

        Uses regex patterns to reliably extract all dates, then shifts them
        using the shift_datetime tool.

        Args:
            content: Original text content

        Returns:
            Tuple of (content with shifted dates, list of shifts made)
        """
        modified_content = content
        all_shifts: List[DateTimeShift] = []

        # Extract all dates using regex
        print("  Extracting dates using pattern matching...")
        found_dates = self._extract_dates_with_regex(content)
        print(f"  Found {len(found_dates)} unique date patterns to process")

        if not found_dates:
            return modified_content, all_shifts

        # Track shifted values to avoid calling the tool multiple times for the same date
        shifted_cache: Dict[str, str] = {}

        # Process each unique date
        for date_str in found_dates:
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
                    print(f"    Shifted: {date_str} -> {shifted_value}")

                except Exception as e:
                    print(f"    Error shifting {date_str}: {e}")
                    shifted_cache[date_str] = date_str  # Keep original
                    continue

            # Apply shift to all occurrences if value changed
            if shifted_value != date_str:
                # Count occurrences
                count = modified_content.count(date_str)
                if count > 0:
                    modified_content = modified_content.replace(date_str, shifted_value)

                    all_shifts.append(DateTimeShift(
                        original_value=date_str,
                        shifted_value=shifted_value,
                        context=f"Found {count} occurrence(s)"
                    ))

        print(f"  Applied {len(all_shifts)} unique date shifts")
        return modified_content, all_shifts

    def _phase2_anonymize_pii(self, content: str) -> Tuple[str, int]:
        """
        Phase 2: Anonymize all PII using agentic tool-calling with redact_text.

        The LLM identifies PII and calls the redact_text tool for each item found.
        Redactions are applied in real-time as the tools are called.

        Args:
            content: Text content (with dates already shifted)

        Returns:
            Tuple of (anonymized content, number of redactions applied)
        """
        modified_content = content
        total_redactions = 0

        # For very long texts, process in chunks
        max_chunk_size = 8000

        if len(content) <= max_chunk_size:
            # Process entire content at once
            modified_content, total_redactions = self._anonymize_chunk(content, 0)
        else:
            # Process in chunks with overlap to avoid missing PII at boundaries
            chunks = self._split_into_chunks(content, max_chunk_size)
            print(f"  Processing {len(chunks)} chunks...")

            for chunk_num, (chunk_start, chunk_text) in enumerate(chunks):
                print(f"  Chunk {chunk_num + 1}/{len(chunks)} (chars {chunk_start}-{chunk_start + len(chunk_text)})")

                # Find PII in this chunk
                _, chunk_redactions = self._anonymize_chunk(chunk_text, chunk_start)
                total_redactions += chunk_redactions

        # Re-run on full text to apply all redactions consistently
        if total_redactions > 0:
            modified_content, _ = self._anonymize_chunk(modified_content, 0)

        return modified_content, total_redactions

    def _split_into_chunks(self, content: str, max_size: int) -> List[Tuple[int, str]]:
        """Split content into overlapping chunks for processing."""
        chunks = []
        overlap = 500  # Characters of overlap between chunks

        start = 0
        while start < len(content):
            end = min(start + max_size, len(content))

            # Try to break at a paragraph or sentence boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind('\n\n', start + max_size - 500, end)
                if para_break > start:
                    end = para_break
                else:
                    # Look for sentence break
                    sent_break = content.rfind('. ', start + max_size - 200, end)
                    if sent_break > start:
                        end = sent_break + 1

            chunks.append((start, content[start:end]))
            start = end - overlap if end < len(content) else end

        return chunks

    def _anonymize_chunk(self, chunk: str, chunk_start: int) -> Tuple[str, int]:
        """
        Anonymize a chunk of text using LLM with redact_text tool.

        Args:
            chunk: Text chunk to anonymize
            chunk_start: Starting position of chunk in original text

        Returns:
            Tuple of (anonymized chunk, number of redactions)
        """
        modified_chunk = chunk
        redactions = 0

        prompt = f"""You are a PII anonymization agent. Analyze this medical text and redact ALL Personal Identifiable Information (PII).

CRITICAL: You MUST scan the ENTIRE document, even if it's very long. Do NOT skip any sections!
If the document has 100+ lines, you MUST check ALL of them for PIIs.

IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

=== TEXT TO ANALYZE ===
{chunk}
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

Example calls:
  redact_text(text_to_redact="John Smith")
  redact_text(text_to_redact="Dr. Jane Wilson")
  redact_text(text_to_redact="555-123-4567")
  redact_text(text_to_redact="123 Main Street")

  Redaction examples:
    - "<subject_id>: 10005749" → "<subject_id>: ********" (tag and colon preserved)
    - "45790175.dat 16 200.0(0)/mV 16 0 19 3475 0 I" → "********.dat 16 200.0(0)/mV 16 0 19 3475 0 I" (metadata preserved, ID replaced)

"""

        messages = [HumanMessage(content=prompt)]

        # Agentic loop
        max_iterations = 50
        iteration = 0

        def invoke_llm_with_retry(msgs):
            """Invoke LLM with retry logic."""
            return retry_with_backoff(
                lambda: self.llm_anonymize.invoke(msgs),
                config=self.retry_config,
                on_retry=create_retry_callback(prefix="    [LLM] "),
            )

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

                        # Get the redacted version (asterisks)
                        result = redact_text.invoke(tool_args)

                        if "[REDACT_FAILED" not in result and text_to_redact:
                            # Apply to the chunk
                            if text_to_redact in modified_chunk:
                                modified_chunk = modified_chunk.replace(text_to_redact, result)
                                redactions += 1
                                display_text = text_to_redact[:40] + "..." if len(text_to_redact) > 40 else text_to_redact
                                print(f"    Redacted: '{display_text}'")

                        messages.append(ToolMessage(
                            content=f"Redacted: '{text_to_redact}' -> '{result}'",
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                print(f"    Error during anonymization: {e}")
                break

        return modified_chunk, redactions

    def _phase3_verify_and_fix(
        self,
        original_content: str,
        anonymized_content: str
    ) -> Tuple[str, int]:
        """
        Phase 3: Verification agent checks the anonymized output and fixes any issues.

        The agent compares original and anonymized data to identify:
        1. Unshifted dates (dates that appear in both original and anonymized)
        2. Unredacted PII (names, IDs, etc. that weren't anonymized)
        3. Over-redaction (non-PII that was incorrectly redacted)
        4. Unshifted years (e.g. "2024" or "2024-2026" that should be shifted)

        Args:
            original_content: Original text content (before anonymization)
            anonymized_content: Anonymized text content

        Returns:
            Tuple of (fixed content, number of fixes applied)
        """
        modified_content = anonymized_content
        total_fixes = 0

        # For very long texts, process in chunks
        max_chunk_size = 4000

        if len(original_content) <= max_chunk_size:
            modified_content, total_fixes = self._verify_chunk(
                original_content, anonymized_content, 0
            )
        else:
            # Process in chunks
            orig_chunks = self._split_into_chunks(original_content, max_chunk_size)
            anon_chunks = self._split_into_chunks(anonymized_content, max_chunk_size)

            # Use minimum number of chunks
            num_chunks = min(len(orig_chunks), len(anon_chunks))
            print(f"  Verifying {num_chunks} chunks...")

            for chunk_num in range(num_chunks):
                _, orig_chunk = orig_chunks[chunk_num]
                _, anon_chunk = anon_chunks[chunk_num]

                print(f"  Verifying chunk {chunk_num + 1}/{num_chunks}")
                _, chunk_fixes = self._verify_chunk(orig_chunk, anon_chunk, chunk_num)
                total_fixes += chunk_fixes

        return modified_content, total_fixes

    def _verify_chunk(
        self,
        original_chunk: str,
        anonymized_chunk: str,
        chunk_num: int
    ) -> Tuple[str, int]:
        """
        Verify and fix a chunk of text.

        Args:
            original_chunk: Original text chunk
            anonymized_chunk: Anonymized text chunk
            chunk_num: Chunk number for logging

        Returns:
            Tuple of (fixed chunk, number of fixes)
        """
        modified_chunk = anonymized_chunk
        fixes = 0

        # Show full content for verification - no truncation

        prompt = f"""You are a verification agent for medical data anonymization.
Compare the ORIGINAL and ANONYMIZED text below and identify any issues.

CRITICAL: You MUST check the ENTIRE document carefully!
Do NOT stop at the first few lines - scan ALL the way to the end.

=== ORIGINAL TEXT ===
{original_chunk}

=== ANONYMIZED TEXT ===
{anonymized_chunk}

Time offset used: {self.time_offset_days} days

You have THREE tools available:
1. shift_datetime - to fix unshifted dates
2. redact_text - to redact PII that was missed
3. restore_text - to fix over-redaction (restore incorrectly redacted content)

Your tasks (BE THOROUGH - check EVERY line of long documents):

1. CHECK FOR UNSHIFTED DATES: Look for dates in ANONYMIZED that are identical to ORIGINAL.
   All dates should be shifted by {self.time_offset_days} days.
   → Use shift_datetime(datetime_str, offset_days={self.time_offset_days}) to fix

2. CHECK FOR UNREDACTED PII: Look for PII that appears UNCHANGED:
   - Patient names, doctor names, staff names
   - Phone numbers, fax numbers, email addresses
   - Specific addresses or facility names
   → Use redact_text(text_to_redact="the PII text") to fix

3. CHECK FOR OVER-REDACTION: Look for asterisks that replaced NON-PII content:
   - Medical terminology (diabetes, hypertension, pneumonia, etc.)
   - Medication names (metformin, lisinopril, aspirin, etc.)
   - Procedure names (colonoscopy, MRI, CT scan, etc.)
   - Generic locations (EMERGENCY ROOM, ICU, HOME, etc.)
   → Use restore_text(redacted_text="*****", original_text="original term") to fix

IMPORTANT:
- Compare ORIGINAL vs ANONYMIZED to identify issues
- Only redact actual PII, not medical content
- Restore any medical terms that were incorrectly redacted
- Check the ENTIRE document, not just the first page

Call the appropriate tool for each issue you find. When done, summarize your findings.
"""

        messages = [HumanMessage(content=prompt)]

        # Agentic loop for verification
        max_iterations = 30
        iteration = 0

        def invoke_verify_with_retry(msgs):
            """Invoke verification LLM with retry logic."""
            return retry_with_backoff(
                lambda: self.llm_verify.invoke(msgs),
                config=self.retry_config,
                on_retry=create_retry_callback(prefix="    [Verify] "),
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
                            if original_date in modified_chunk:
                                modified_chunk = modified_chunk.replace(original_date, result)
                                fixes += 1
                                print(f"    Fixed date: {original_date} -> {result}")

                        messages.append(ToolMessage(
                            content=f"Date shifted: {original_date} -> {result}",
                            tool_call_id=tool_call["id"]
                        ))

                    elif tool_name == "redact_text":
                        text_to_redact = tool_args.get("text_to_redact", "")
                        result = redact_text.invoke(tool_args)

                        if "[REDACT_FAILED" not in result:
                            if text_to_redact in modified_chunk:
                                modified_chunk = modified_chunk.replace(text_to_redact, result)
                                fixes += 1
                                print(f"    Fixed PII: '{text_to_redact}' -> '{result}'")

                        messages.append(ToolMessage(
                            content=f"Text redacted: '{text_to_redact}' -> '{result}'",
                            tool_call_id=tool_call["id"]
                        ))

                    elif tool_name == "restore_text":
                        redacted_text = tool_args.get("redacted_text", "")
                        original_text = tool_args.get("original_text", "")
                        result = restore_text.invoke(tool_args)

                        if "[RESTORE_FAILED" not in result:
                            if redacted_text in modified_chunk:
                                modified_chunk = modified_chunk.replace(redacted_text, result)
                                fixes += 1
                                print(f"    Restored: '{redacted_text}' -> '{result}'")

                        messages.append(ToolMessage(
                            content=f"Text restored: '{redacted_text}' -> '{result}'",
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                print(f"    Verification error: {e}")
                break

        return modified_chunk, fixes

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
                "processing_method": "agentic_text_anonymization",
                "time_offset_days": self.time_offset_days,
                "total_dates_shifted": len(dates_shifted),
                "total_pii_redactions": pii_redactions_count
            },
            "phase1_time_shifts": [
                {
                    "original": d.original_value,
                    "shifted": d.shifted_value,
                    "context": d.context
                }
                for d in dates_shifted
            ]
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
