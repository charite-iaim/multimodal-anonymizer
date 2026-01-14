"""
Agentic text file processor using LLM with tool-calling for anonymization.

This processor uses a two-step agentic approach:
1. LLM identifies and shifts dates/times using the shift_datetime tool
2. LLM anonymizes all other PII (but not the already-shifted dates)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import random

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..tools.time_shift_tool import shift_datetime


class DateTimeFound(BaseModel):
    """A date/time found in the text."""
    original_value: str = Field(description="The original date/time string as it appears in the text")
    shifted_value: str = Field(description="The date/time after shifting")
    context: str = Field(description="Brief context where this date was found")


class TimeShiftResult(BaseModel):
    """Result of the time shifting step."""
    content_with_shifted_dates: str = Field(description="The complete text with all dates shifted")
    dates_shifted: List[DateTimeFound] = Field(description="List of dates that were shifted")


class TextAnonymization(BaseModel):
    """Anonymization result for text content."""
    original_text: str = Field(description="Original text snippet containing PHI")
    anonymized_text: str = Field(description="Text with PHI replaced by asterisks")
    phi_category: str = Field(description="Category of PHI (name, address, id, phone, etc.)")


class AnonymizationResult(BaseModel):
    """Final anonymization result."""
    anonymized_content: str = Field(description="Complete anonymized text content")
    anonymizations: List[TextAnonymization] = Field(description="List of PHI items that were anonymized")


class AgenticTextProcessor(FileProcessor):
    """
    Agentic processor for text files using LLM with tool-calling.

    This processor implements a two-phase approach:
    1. Time-Shift Phase: LLM uses the shift_datetime tool to find and shift all dates
    2. Anonymization Phase: LLM anonymizes all other PII (names, addresses, IDs, etc.)
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

        # Initialize LLM with tools for phase 1 (time shifting)
        self.llm_with_tools = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
            timeout=300,
        ).bind_tools([shift_datetime])

        # Initialize LLM with structured output for phase 2 (anonymization)
        self.llm_anonymize = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
            timeout=300,
        ).with_structured_output(AnonymizationResult)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a text file (.txt or .hea ECG header)."""
        return file_path.suffix.lower() in [".txt", ".hea"]

    def extract_content(self, file_path: Path) -> str:
        """Extract text content as string."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize text file using agentic LLM approach.

        Steps:
        1. Read text file
        2. Phase 1: Use LLM with tools to find and shift all dates/times
        3. Phase 2: Use LLM to anonymize all other PII
        4. Save anonymized text

        Args:
            input_path: Path to input text file
            output_path: Path to save anonymized text file
        """
        # Convert to Path if string
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        output_path = Path(output_path) if isinstance(output_path, str) else output_path

        print(f"Processing (Agentic): {input_path.name}")
        print(f"Time offset: {self.time_offset_days} days")

        # Step 1: Read text file
        content = self.extract_content(input_path)
        print(f"Read {len(content)} characters")

        if not content.strip():
            print("Empty text file, saving as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return

        # Step 2: Phase 1 - Time shifting with tools
        print("\n=== Phase 1: Time Shifting ===")
        shifted_content, dates_shifted = self._phase1_shift_times(content)
        print(f"Shifted {len(dates_shifted)} date/time values")

        # Step 3: Phase 2 - Anonymize other PII
        print("\n=== Phase 2: PII Anonymization ===")
        anonymization_result = self._phase2_anonymize_pii(shifted_content)
        print(f"Anonymized {len(anonymization_result.anonymizations)} PHI items")

        # Step 4: Save anonymized text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(anonymization_result.anonymized_content)
        print(f"Saved anonymized text to: {output_path}")

        # Step 5: Save JSON with details (if debug mode)
        if self.config.save_debug_files:
            json_output_path = output_path.with_suffix('.json')
            self._save_json_output(
                dates_shifted,
                anonymization_result,
                input_path,
                output_path,
                json_output_path
            )
            print(f"Saved anonymization details to: {json_output_path}")

    def _phase1_shift_times(self, content: str) -> tuple[str, List[DateTimeFound]]:
        """
        Phase 1: Use LLM with tool-calling to find and shift all dates/times.

        The LLM will:
        1. Analyze the text to find all date/time values
        2. Call the shift_datetime tool for each one
        3. Return the modified text with all dates shifted

        Args:
            content: Original text content

        Returns:
            Tuple of (content with shifted dates, list of shifts made)
        """
        prompt = f"""Analyze this medical document and find ALL date and time values.

For EACH date/time you find, use the shift_datetime tool to shift it by {self.time_offset_days} days.

Date/time formats to look for:
- ISO dates: 2024-03-15, 2024-03-15T14:30:00
- European dates: 15.03.2024, 15/03/2024
- US dates: 03/15/2024, March 15, 2024
- Dates with times: 2024-03-15 14:30
- Relative dates that include specific dates

IMPORTANT:
- Find and shift ALL dates, including admission dates, discharge dates, birth dates, procedure dates
- Call shift_datetime for each date found
- Keep track of which dates you've shifted

=== DOCUMENT ===
{content[:8000]}
"""

        messages = [HumanMessage(content=prompt)]
        dates_shifted = []
        modified_content = content

        # Agentic loop: keep calling until no more tool calls
        max_iterations = 20  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            # Check if there are tool calls
            if not response.tool_calls:
                print(f"  Phase 1 complete after {iteration} iterations")
                break

            # Process each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "shift_datetime":
                    original = tool_args.get("datetime_str", "")
                    offset = tool_args.get("offset_days", self.time_offset_days)

                    # Execute the tool
                    result = shift_datetime.invoke(tool_args)

                    print(f"  Shifted: {original} → {result}")

                    # Track the shift
                    dates_shifted.append(DateTimeFound(
                        original_value=original,
                        shifted_value=result,
                        context="Found by LLM"
                    ))

                    # Apply to content
                    if "[SHIFT_FAILED]" not in result:
                        modified_content = modified_content.replace(original, result, 1)

                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=f"Date shifted: {original} → {result}",
                        tool_call_id=tool_call["id"]
                    ))

        return modified_content, dates_shifted

    def _phase2_anonymize_pii(self, content: str) -> AnonymizationResult:
        """
        Phase 2: Anonymize all PII except dates (which are already shifted).

        Args:
            content: Text content with dates already shifted

        Returns:
            AnonymizationResult with anonymized content
        """
        prompt = f"""Analyze this medical document and anonymize all Personal Identifiable Information (PII).

IMPORTANT: Dates and times have ALREADY been anonymized (shifted). Do NOT redact dates!

PII categories to redact (replace with asterisks):
- name: Patient names, physician names, doctor names, family member names
- address: Physical addresses, street addresses, facility names, hospital names
- id: Patient IDs, medical record numbers, unit numbers, any numeric identifiers
- phone: Phone numbers
- fax: Fax numbers
- email: Email addresses
- age: Patient ages

DO NOT redact:
- Dates (they are already anonymized by shifting)
- Times
- Medical terminology, procedures, medications
- Lab values, measurements

Instructions:
1. Replace EVERY CHARACTER of PHI (including spaces, hyphens) with asterisks (*)
2. Preserve document structure and formatting
3. Keep all medical information intact

Example redactions:
- "John Doe" → "********"
- "Dr. Jane Smith" → "**************"
- "123-456-7890" → "************"
- "Patient ID: 12345678" → "Patient ID: ********"

=== DOCUMENT TO ANONYMIZE ===
{content[:10000]}
"""

        message = HumanMessage(content=prompt)

        try:
            result: AnonymizationResult = self.llm_anonymize.invoke([message])

            # Print details
            print(f"\nAnonymization details:")
            for anon in result.anonymizations:
                print(f"  {anon.phi_category}: '{anon.original_text}' → '{anon.anonymized_text}'")

            return result

        except Exception as e:
            print(f"Error during anonymization: {e}")
            import traceback
            traceback.print_exc()
            return AnonymizationResult(
                anonymized_content=content,
                anonymizations=[]
            )

    def _save_json_output(
        self,
        dates_shifted: List[DateTimeFound],
        anonymization_result: AnonymizationResult,
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
            },
            "phase1_time_shifts": [
                {
                    "original": d.original_value,
                    "shifted": d.shifted_value,
                    "context": d.context
                }
                for d in dates_shifted
            ],
            "phase2_anonymizations": [
                {
                    "phi_category": anon.phi_category,
                    "original_text": anon.original_text,
                    "anonymized_text": anon.anonymized_text
                }
                for anon in anonymization_result.anonymizations
            ]
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
