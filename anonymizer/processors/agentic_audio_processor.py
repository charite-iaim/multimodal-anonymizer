"""
Agentic audio file processor for voice data anonymization.

This processor handles .wav and .mp3 audio files through a three-step pipeline:
1. Transcription: Use local Whisper model to convert speech to text
2. Anonymization: Use LLM to remove PII (names, dates, etc.) from transcript
3. Synthesis: Use local pyttsx3 TTS to convert anonymized text back to speech

The output is saved in the same format as the original file.
All processing runs locally (except for LLM anonymization which uses configured provider).
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import random

from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..llm_factory import create_chat_llm
from ..tools.time_shift_tool import shift_datetime, redact_text
from ..retry_utils import retry_with_backoff, RetryConfig, create_retry_callback
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG

# Local Whisper import
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Local pyttsx3 TTS import
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Audio format conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class AgenticAudioProcessor(FileProcessor):
    """
    Agentic processor for audio files (.wav, .mp3) using local Whisper + LLM + local TTS.

    Pipeline:
    1. Local Whisper model transcribes audio to text
    2. LLM anonymizes the transcript (removes PII, shifts dates)
    3. Local pyttsx3 TTS converts anonymized text back to speech
    4. Audio is saved in the original format
    
    All processing runs locally except for LLM anonymization.
    
    pyttsx3 uses the system's built-in speech synthesis:
    - macOS: NSSpeechSynthesizer
    - Windows: SAPI5
    - Linux: espeak
    """

    # Supported audio formats
    SUPPORTED_EXTENSIONS = [".wav", ".mp3"]

    # Available Whisper model sizes (from smallest/fastest to largest/most accurate)
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(
        self,
        config: AnonymizerConfig,
        time_offset_days: Optional[int] = None,
        prompt_config: Optional[PromptConfig] = None,
        whisper_model_size: str = "large",
        tts_rate: int = 150,
        tts_voice_id: Optional[str] = None,
    ):
        """
        Initialize agentic audio processor with fully local speech processing.

        Args:
            config: Configuration object with LLM settings
            time_offset_days: Fixed offset for time shifting. If None, a random offset is generated.
            prompt_config: Custom prompt configuration. If None, uses default prompts.
            whisper_model_size: Local Whisper model size (tiny, base, small, medium, large)
            tts_rate: Speech rate in words per minute (default: 150)
            tts_voice_id: System voice ID to use. If None, uses system default.
                         Use list_available_voices() to see available voices.
        """
        super().__init__(config)

        # Check dependencies
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Local Whisper library is required for audio transcription. "
                "Install it with: pip install -U openai-whisper"
            )
        if not PYTTSX3_AVAILABLE:
            raise ImportError(
                "pyttsx3 library is required for local speech synthesis. "
                "Install it with: pip install pyttsx3"
            )
        if not PYDUB_AVAILABLE:
            raise ImportError(
                "pydub library is required for audio format conversion. "
                "Install it with: pip install pydub\n"
                "Also ensure ffmpeg is installed on your system."
            )

        # Prompt configuration
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG

        # Generate random offset if not provided (between -365 and +365 days)
        if time_offset_days is None:
            self.time_offset_days = random.randint(-365, 365)
        else:
            self.time_offset_days = time_offset_days

        # Whisper configuration
        self.whisper_model_size = whisper_model_size if whisper_model_size in self.WHISPER_MODELS else "large"
        
        # Load local Whisper model
        print(f"Loading Whisper model: {self.whisper_model_size}...")
        self.whisper_model = whisper.load_model(self.whisper_model_size)
        print(f"Whisper model loaded successfully")

        # TTS configuration
        self.tts_rate = tts_rate
        self.tts_voice_id = tts_voice_id
        
        # Test TTS engine initialization
        print("Initializing TTS engine...")
        self._test_tts_engine()
        print("TTS engine ready")

        # Configure retry settings for LLM calls
        self.retry_config = RetryConfig(
            max_retries=3,
            initial_delay=2.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Initialize LLM with tools for anonymization
        self.llm_anonymize = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[shift_datetime, redact_text],
        )

    def _test_tts_engine(self):
        """Test that TTS engine can be initialized."""
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f"  Available system voices: {len(voices)}")
        engine.stop()

    def _create_tts_engine(self):
        """Create a new TTS engine instance with configured settings."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.tts_rate)
        
        if self.tts_voice_id:
            engine.setProperty('voice', self.tts_voice_id)
        
        return engine

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported audio file (.wav or .mp3)."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract_content(self, file_path: Path) -> str:
        """
        Extract content from audio file by transcribing it.

        Args:
            file_path: Path to the audio file

        Returns:
            Transcribed text from the audio
        """
        return self._transcribe_audio(file_path)

    def anonymize(self, input_path: Path, output_path: Path, verify: bool = True) -> None:
        """
        Anonymize audio file using local Whisper + LLM + local TTS pipeline.

        Steps:
        1. Transcribe audio to text using local Whisper model
        2. Anonymize text using LLM (redact PII, shift dates)
        3. Convert anonymized text back to speech using local pyttsx3 TTS
        4. Save in the original audio format

        Args:
            input_path: Path to input audio file
            output_path: Path to save anonymized audio file
            verify: Whether to run verification phase (default: True)
        """
        # Convert to Path if string
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        output_path = Path(output_path) if isinstance(output_path, str) else output_path

        original_format = input_path.suffix.lower()
        print(f"Processing audio (Agentic): {input_path.name}")
        print(f"Time offset: {self.time_offset_days} days")
        print(f"Original format: {original_format}")

        # Step 1: Transcribe audio to text using local Whisper
        print("\n=== Step 1: Transcription (Local Whisper) ===")
        transcript = self._transcribe_audio(input_path)
        print(f"Transcribed {len(transcript)} characters")

        if not transcript.strip():
            print("Empty transcript, creating silent audio placeholder")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_silent_audio(output_path, original_format)
            return

        print(f"Transcript preview: {transcript[:200]}...")

        # Step 2: Anonymize the transcript
        print("\n=== Step 2: Text Anonymization (LLM) ===")
        anonymized_text, stats = self._anonymize_transcript(transcript, verify=verify)
        print(f"Anonymized text ({stats['dates_shifted']} dates shifted, {stats['pii_redacted']} PII redacted)")
        print(f"Anonymized preview: {anonymized_text[:200]}...")

        # Step 3: Convert anonymized text to speech using local TTS
        print("\n=== Step 3: Speech Synthesis (Local pyttsx3 TTS) ===")
        print(f"Speech rate: {self.tts_rate} words/min")
        tts_audio_path = self._synthesize_speech(anonymized_text)

        # Step 4: Convert to original format and save
        print("\n=== Step 4: Format Conversion & Save ===")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._convert_and_save(tts_audio_path, output_path, original_format)
        print(f"Saved anonymized audio to: {output_path}")

        # Clean up temp file
        if tts_audio_path.exists():
            tts_audio_path.unlink()

        # Save metadata if debug mode
        if self.config.save_debug_files:
            self._save_debug_metadata(
                input_path=input_path,
                output_path=output_path,
                original_transcript=transcript,
                anonymized_transcript=anonymized_text,
                stats=stats,
            )

    def _transcribe_audio(self, audio_path: Path) -> str:
        """
        Transcribe audio to text using local Whisper model.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text
        """
        print(f"  Transcribing with local Whisper ({self.whisper_model_size} model)...")

        # Whisper can handle both wav and mp3 directly
        result = self.whisper_model.transcribe(str(audio_path))
        
        transcript = result.get("text", "").strip()
        
        # Log detected language if available
        if "language" in result:
            print(f"  Detected language: {result['language']}")
        
        return transcript

    def _anonymize_transcript(
        self, transcript: str, verify: bool = True
    ) -> Tuple[str, dict]:
        """
        Anonymize transcript text using LLM with tool-calling.

        Args:
            transcript: Original transcript text
            verify: Whether to run verification pass

        Returns:
            Tuple of (anonymized text, stats dict)
        """
        stats = {"dates_shifted": 0, "pii_redacted": 0}

        # Phase 1: Extract and shift dates
        print("  Phase 1: Shifting dates...")
        modified_text, dates_shifted = self._shift_dates_in_text(transcript)
        stats["dates_shifted"] = dates_shifted

        # Phase 2: Redact PII using LLM
        print("  Phase 2: Redacting PII...")
        modified_text, pii_count = self._redact_pii(modified_text)
        stats["pii_redacted"] = pii_count

        # Phase 3: Verification (optional)
        if verify:
            print("  Phase 3: Verification...")
            modified_text, verify_fixes = self._verify_and_fix(
                original_text=transcript,
                anonymized_text=modified_text
            )
            stats["verification_fixes"] = verify_fixes

        return modified_text, stats

    def _shift_dates_in_text(self, text: str) -> Tuple[str, int]:
        """
        Find and shift all dates in text using regex + shift_datetime tool.

        Args:
            text: Input text

        Returns:
            Tuple of (text with shifted dates, count of shifts)
        """
        import re

        modified_text = text
        shift_count = 0

        # Date patterns to find
        patterns = [
            r'\b(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{2}/\d{2}/\d{4})\b',
            r'\b(\d{2}\.\d{2}\.\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b',
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
        ]

        found_dates = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_dates.update(matches)

        # Shift each unique date
        shifted_cache = {}
        for date_str in sorted(found_dates, key=len, reverse=True):
            if date_str in shifted_cache:
                continue

            try:
                result = shift_datetime.invoke({
                    "datetime_str": date_str,
                    "offset_days": self.time_offset_days
                })

                if "[SHIFT_FAILED]" not in result:
                    shifted_cache[date_str] = result
                    count = modified_text.count(date_str)
                    if count > 0:
                        modified_text = modified_text.replace(date_str, result)
                        shift_count += count
                        print(f"    Shifted: {date_str} -> {result}")
            except Exception as e:
                print(f"    Skip date {date_str}: {e}")

        return modified_text, shift_count

    def _redact_pii(self, text: str) -> Tuple[str, int]:
        """
        Use LLM with redact_text tool to anonymize PII in text.

        Args:
            text: Input text (dates already shifted)

        Returns:
            Tuple of (anonymized text, count of redactions)
        """
        modified_text = text
        redaction_count = 0

        # Build prompt for audio transcript anonymization
        prompt = f"""You are an anonymization specialist for medical audio transcripts.

Your task is to identify and redact all Personal Identifiable Information (PII) in this transcript.

TRANSCRIPT:
{text}

Use the `redact_text` tool to replace each PII item with appropriate placeholders.

PII to redact includes:
- Patient names, doctor names, staff names
- Hospital names, clinic names, facility names
- Street addresses, city names, specific locations
- Phone numbers, fax numbers
- Email addresses
- Social Security numbers, medical record numbers, patient IDs
- Dates of birth (if specific dates, shift them; if "date of birth" as phrase, keep as placeholder)
- Ages (if combined with other identifying information)
- Any other information that could identify a specific individual

IMPORTANT:
- Keep medical terms, diagnoses, treatments, medications intact
- Replace names with [NAME], [DOCTOR], [HOSPITAL], etc.
- Replace specific numbers/IDs with [ID], [PHONE], [MRN], etc.
- Preserve sentence structure and natural language flow

Call the `redact_text` tool for each PII item found. When done, respond with "ANONYMIZATION COMPLETE".
"""

        messages = [HumanMessage(content=prompt)]

        # Agentic loop
        max_iterations = 50
        iteration = 0

        def invoke_with_retry(msgs):
            return retry_with_backoff(
                lambda: self.llm_anonymize.invoke(msgs),
                config=self.retry_config,
                on_retry=create_retry_callback(prefix="    [LLM] "),
            )

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

                    if tool_name == "redact_text":
                        text_to_redact = tool_args.get("text_to_redact", "")
                        result = redact_text.invoke(tool_args)

                        if "[REDACT_FAILED" not in result and text_to_redact:
                            if text_to_redact in modified_text:
                                modified_text = modified_text.replace(text_to_redact, result)
                                redaction_count += 1
                                display = text_to_redact[:30] + "..." if len(text_to_redact) > 30 else text_to_redact
                                print(f"    Redacted: '{display}'")

                        messages.append(ToolMessage(
                            content=f"Redacted: '{text_to_redact}' -> '{result}'",
                            tool_call_id=tool_call["id"]
                        ))

                    elif tool_name == "shift_datetime":
                        # Handle any additional date shifts
                        date_str = tool_args.get("datetime_str", "")
                        result = shift_datetime.invoke(tool_args)

                        if "[SHIFT_FAILED]" not in result and date_str in modified_text:
                            modified_text = modified_text.replace(date_str, result)
                            print(f"    Shifted (via LLM): {date_str} -> {result}")

                        messages.append(ToolMessage(
                            content=f"Date shifted: {date_str} -> {result}",
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                print(f"    Error during PII redaction: {e}")
                break

        return modified_text, redaction_count

    def _verify_and_fix(self, original_text: str, anonymized_text: str) -> Tuple[str, int]:
        """
        Verification phase to catch any missed PII.

        Args:
            original_text: Original transcript
            anonymized_text: Anonymized transcript

        Returns:
            Tuple of (verified/fixed text, number of fixes)
        """
        modified_text = anonymized_text
        fixes = 0

        prompt = f"""You are a verification agent for medical transcript anonymization.

Compare the original and anonymized transcripts below. Look for ANY remaining PII that was missed.

ORIGINAL TRANSCRIPT:
{original_text}

ANONYMIZED TRANSCRIPT:
{anonymized_text}

TIME OFFSET USED: {self.time_offset_days} days

Your task:
1. Check if any names, locations, IDs, or other PII remain in the anonymized version
2. Check if any dates were not properly shifted
3. Use `redact_text` or `shift_datetime` tools to fix any issues found

When verification is complete and no more issues found, respond with "VERIFICATION COMPLETE".
"""

        messages = [HumanMessage(content=prompt)]
        max_iterations = 30
        iteration = 0

        def invoke_with_retry(msgs):
            return retry_with_backoff(
                lambda: self.llm_anonymize.invoke(msgs),
                config=self.retry_config,
                on_retry=create_retry_callback(prefix="    [Verify] "),
            )

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

                    if tool_name == "redact_text":
                        text_to_redact = tool_args.get("text_to_redact", "")
                        result = redact_text.invoke(tool_args)

                        if "[REDACT_FAILED" not in result and text_to_redact in modified_text:
                            modified_text = modified_text.replace(text_to_redact, result)
                            fixes += 1
                            print(f"    Fixed PII: '{text_to_redact}' -> '{result}'")

                        messages.append(ToolMessage(
                            content=f"Redacted: '{text_to_redact}' -> '{result}'",
                            tool_call_id=tool_call["id"]
                        ))

                    elif tool_name == "shift_datetime":
                        date_str = tool_args.get("datetime_str", "")
                        result = shift_datetime.invoke(tool_args)

                        if "[SHIFT_FAILED]" not in result and date_str in modified_text:
                            modified_text = modified_text.replace(date_str, result)
                            fixes += 1
                            print(f"    Fixed date: {date_str} -> {result}")

                        messages.append(ToolMessage(
                            content=f"Date shifted: {date_str} -> {result}",
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                print(f"    Verification error: {e}")
                break

        return modified_text, fixes

    def _synthesize_speech(self, text: str) -> Path:
        """
        Convert text to speech using local pyttsx3 TTS.

        Args:
            text: Text to convert to speech

        Returns:
            Path to the generated audio file (WAV format)
        """
        print(f"  Synthesizing speech ({len(text)} characters)...")

        output_path = Path(tempfile.mktemp(suffix=".wav"))
        
        # Create TTS engine
        engine = self._create_tts_engine()
        
        try:
            # Save speech to file
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
        finally:
            engine.stop()
        
        # Verify the file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("TTS failed to generate audio file")
        
        print(f"  Generated audio: {output_path.stat().st_size / 1024:.1f} KB")
        return output_path

    def _convert_and_save(self, source_path: Path, output_path: Path, target_format: str) -> None:
        """
        Convert audio to target format and save.

        Args:
            source_path: Path to source audio (from TTS - may be WAV or AIFF depending on platform)
            output_path: Path to save converted audio
            target_format: Target format (.wav or .mp3)
        """
        # Use from_file() for auto-detection since pyttsx3 generates different formats
        # on different platforms (AIFF on macOS, WAV on Windows/Linux)
        audio = AudioSegment.from_file(str(source_path))

        if target_format == ".wav":
            audio.export(str(output_path), format="wav")
        elif target_format == ".mp3":
            audio.export(str(output_path), format="mp3")
        else:
            # Default to wav
            audio.export(str(output_path), format="wav")

    def _create_silent_audio(self, output_path: Path, format: str, duration_ms: int = 1000) -> None:
        """Create a silent audio file as placeholder for empty transcripts."""
        silent = AudioSegment.silent(duration=duration_ms)

        if format == ".wav":
            silent.export(str(output_path), format="wav")
        else:
            silent.export(str(output_path), format="mp3")

    def _save_debug_metadata(
        self,
        input_path: Path,
        output_path: Path,
        original_transcript: str,
        anonymized_transcript: str,
        stats: dict,
    ) -> None:
        """Save debug metadata as JSON file."""
        json_path = output_path.with_suffix(".json")

        metadata = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "agentic_audio_anonymization_local",
                "time_offset_days": self.time_offset_days,
                "whisper_model_size": self.whisper_model_size,
                "tts_rate": self.tts_rate,
                "tts_voice_id": self.tts_voice_id,
            },
            "stats": stats,
            "original_transcript": original_transcript,
            "anonymized_transcript": anonymized_transcript,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  Saved debug metadata to: {json_path}")

    @staticmethod
    def list_available_voices() -> list:
        """
        List all available system TTS voices.
        
        Returns:
            List of voice information dictionaries with 'id' and 'name' keys
        """
        if not PYTTSX3_AVAILABLE:
            print("pyttsx3 not installed. Install with: pip install pyttsx3")
            return []
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        voice_list = []
        for voice in voices:
            voice_list.append({
                'id': voice.id,
                'name': voice.name,
                'languages': getattr(voice, 'languages', []),
                'gender': getattr(voice, 'gender', None),
            })
        
        engine.stop()
        return voice_list
