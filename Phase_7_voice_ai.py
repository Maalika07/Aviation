from __future__ import annotations

import argparse
import os
import sys
import time
import wave
import tempfile
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import whisper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 8
SILENCE_THRESH = 0.015
SILENCE_SECS = 2.0

AIRPORT_KEYWORDS = {
    "flight": ["flight", "plane", "aircraft", "departure", "arrival"],
    "delay": ["delay", "delayed", "late", "hold", "slow"],
    "gate": ["gate", "stand", "terminal", "pier"],
    "congestion": ["congestion", "congested", "queue", "bottleneck", "crowded"],
    "fuel": ["fuel", "fueling", "refuel", "truck"],
    "weather": ["weather", "storm", "rain", "fog", "wind", "clear"],
    "staff": ["staff", "crew", "ramp", "operator", "personnel"],
    "runway": ["runway", "taxiway", "landing", "takeoff"],
}

class WhisperTranscriber:
    def __init__(self, model_name: str = "base") -> None:
        logger.info(f"Whisper Loading model: {model_name} ...")
        self.model_name = model_name
        self.model = whisper.load_model(model_name)
        device = next(self.model.parameters()).device
        logger.info(f"Whisper Model loaded on {device}")

    def transcribe_file(self, audio_path: str) -> dict:
        logger.info(f"Whisper Transcribing: {audio_path}")
        result = self.model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            fp16=False,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
        )
        text = result["text"].strip()
        lang = result.get("language", "en")
        logger.info(f"Whisper Transcribed ({lang}): '{text}'")
        return result

    def transcribe_array(self, audio_array: np.ndarray) -> str:
        result = self.model.transcribe(
            audio_array.astype(np.float32),
            language="en",
            task="transcribe",
            fp16=False,
            no_speech_threshold=0.6,
        )
        return result["text"].strip()

class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        max_seconds: int = RECORD_SECONDS,
        silence_threshold: float = SILENCE_THRESH,
        silence_seconds: float = SILENCE_SECS,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.silence_threshold = silence_threshold
        self.silence_seconds = silence_seconds
        self._sd_available = self._check_sounddevice()

    @staticmethod
    def _check_sounddevice() -> bool:
        try:
            import sounddevice
            return True
        except (ImportError, OSError):
            logger.warning("Recorder sounddevice not install -- use --file or --text mode.")
            return False

    def record(self) -> Optional[np.ndarray]:
        if not self._sd_available:
            return None

        import sounddevice as sd

        print("\n Listening... (speak now, auto-stops after silence)")
        frames = []
        silence_buf = []
        started = False

        def callback(indata, frame_count, time_info, status):
            nonlocal started
            chunk = indata[:, 0].copy()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > self.silence_threshold:
                started = True
                silence_buf.clear()
            elif started:
                silence_buf.append(chunk)

            if started:
                frames.append(chunk)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=callback,
        ):
            start = time.time()
            while True:
                time.sleep(0.05)
                elapsed = time.time() - start

                silence_dur = len(silence_buf) * 1024 / self.sample_rate
                if started and silence_dur >= self.silence_seconds:
                    print("Silence detected -- processing...")
                    break

                if elapsed >= self.max_seconds:
                    print("Max duration reached -- processing...")
                    break

        if not frames:
            print("No speech detected.")
            return None

        audio = np.concatenate(frames, axis=0)
        logger.info(f"Recorder Captured {len(audio)/self.sample_rate:.1f}s of audio")
        return audio

    def save_wav(self, audio: np.ndarray, path: str) -> None:
        pcm = (audio * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())

class QueryPreprocessor:
    HALLUCINATIONS = [
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "bye bye",
    ]

    NUMBER_WORDS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    }

    def process(self, raw_text: str) -> str:
        text = raw_text.strip()

        for phrase in self.HALLUCINATIONS:
            if phrase in text.lower():
                text = text.lower().replace(phrase, "").strip()

        for word, digit in self.NUMBER_WORDS.items():
            text = text.replace(word, digit)

        question_starters = ["why", "what", "how", "which", "when", "where", "is", "are", "can"]
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in question_starters and not text.endswith("?"):
            text = text.rstrip(".") + "?"

        has_airport_term = any(
            kw in text.lower()
            for keywords in AIRPORT_KEYWORDS.values()
            for kw in keywords
        )
        if not has_airport_term:
            text = f"Airport operations: {text}"

        logger.info(f"Preprocessor Cleaned: '{raw_text}' to '{text}'")
        return text

    def is_valid(self, text: str) -> bool:
        clean = text.strip()
        return len(clean) > 5 and clean.lower() not in ["", ".", "...", "okay", "um", "uh"]

class TextToSpeech:
    def __init__(self, enabled: bool = True, lang: str = "en") -> None:
        self.enabled = enabled
        self.lang = lang
        if enabled:
            try:
                from gtts import gTTS
                logger.info("TTS gTTS ready")
            except (ImportError, OSError):
                logger.warning("TTS gTTS not installed -- pip install gtts")
                self.enabled = False

    def speak(self, text: str, speed: bool = False) -> None:
        if not self.enabled:
            return

        spoken = self._summarize_for_voice(text)

        try:
            from gtts import gTTS
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            tts = gTTS(text=spoken, lang=self.lang, slow=speed)
            tts.save(tmp_path)
            self._play(tmp_path)
            os.unlink(tmp_path)

        except Exception as e:
            logger.warning(f"TTS Playback failed: {e}")

    @staticmethod
    def _play(path: str) -> None:
        try:
            import playsound
            playsound.playsound(path)
            return
        except Exception:
            pass

        if sys.platform == "win32":
            os.system(f'start /wait "" "{path}"')
        elif sys.platform == "darwin":
            os.system(f"afplay '{path}'")
        else:
            os.system(f"mpg123 '{path}' 2>/dev/null || aplay '{path}' 2>/dev/null")

    @staticmethod
    def _summarize_for_voice(text: str, max_chars: int = 400) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines:
            if "rl recommendation" in line.lower() or "action:" in line.lower():
                return line[:max_chars]

        for line in lines:
            if len(line) > 30 and not line.startswith("[") and not line.startswith("="):
                return line[:max_chars]

        return text[:max_chars]

class ResponseFormatter:
    def format_console(self, result: dict, query: str) -> str:
        lines = [
            "",
            "=" * 65,
            " AIRPORT AI -- VOICE QUERY RESPONSE",
            "=" * 65,
            f" Query : {query}",
            f" Flight : {result.get('selected_flight_id', '—')}",
            f" Op ID : {result.get('selected_operation_id', '—')}",
            "",
        ]

        reco = result.get("rl_recommendation", {})
        if reco:
            lines += [
                " RL RECOMMENDATION:",
                f" Action : {reco.get('action', '—')}",
                f" Mode : {reco.get('policy_mode', '—')}",
                f" Reduction : {reco.get('expected_delay_reduction_minutes', 0)} min",
                f" Projected : {reco.get('projected_delay_minutes', '—')} min delay",
                f" Reason : {reco.get('reason', '—')}",
                "",
            ]

        hits = result.get("fused_hits", [])
        if hits:
            lines.append(" TOP RAG+RRF EVIDENCE:")
            for h in hits[:3]:
                lines.append(f" {h.rank}. [{h.retriever}] {h.doc.title[:55]} ({h.score:.4f})")
            lines.append("")

        answer = result.get("final_answer", "")
        if answer:
            lines += [" FINAL ANSWER:", " " + "-" * 61]
            for line in answer.split("\n")[:8]:
                lines.append(f" {line}")

        lines.append("=" * 65)
        return "\n".join(lines)

    def format_voice(self, result: dict) -> str:
        reco = result.get("rl_recommendation", {})
        flight = result.get("selected_flight_id", "the selected flight")
        action = reco.get("action", "hold current plan").replace("_", " ")
        delay = reco.get("projected_delay_minutes", "unknown")
        reason = reco.get("reason", "multiple risk factors detected")

        return (
            f"Analysis complete for flight {flight}. "
            f"Recommended action: {action}. "
            f"Projected delay: {delay} minutes. "
            f"Reason: {reason}."
        )

class AirportVoiceAI:
    def __init__(
        self,
        whisper_model: str = "base",
        enable_tts: bool = True,
        refresh_kb: bool = False,
    ) -> None:
        print("\n" + "=" * 65)
        print(" Airport Voice AI -- Phase 7")
        print(" Stack: Whisper + RAG + RRF + Graph Intelligence + RL")
        print("=" * 65)

        print("\n [1/4] Loading Whisper model...")
        self.transcriber = WhisperTranscriber(model_name=whisper_model)

        print(" [2/4] Initialising audio recorder...")
        self.recorder = AudioRecorder()

        self.preprocessor = QueryPreprocessor()
        self.formatter = ResponseFormatter()

        print(" [3/4] Initialising text-to-speech...")
        self.tts = TextToSpeech(enabled=enable_tts)

        print(" [4/4] Loading agentic workflow (KB + Graph + RL)...")
        self._load_workflow(refresh_kb)

        print("\n Voice AI pipeline ready.\n")

    def _load_workflow(self, refresh_kb: bool) -> None:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from phase_6_agent_workflow import AirportAgenticWorkflow
            self.workflow = AirportAgenticWorkflow(refresh_knowledge=refresh_kb)
            logger.info("VoiceAI Phase 6 workflow loaded")
        except Exception as e:
            logger.warning(f"VoiceAI Phase 6 not available ({e}) -- running in transcribe-only mode.")
            self.workflow = None

    def process_query(self, raw_text: str) -> Optional[dict]:
        query = self.preprocessor.process(raw_text)

        if not self.preprocessor.is_valid(query):
            print("Query too short or unclear. Please speak again.")
            return None

        print(f"\n Query: {query}")

        if self.workflow is None:
            print(" Workflow not loaded -- transcription only mode.")
            return {"final_answer": f"Transcribed: {query}", "query": query}

        print(" Running agentic pipeline...")
        t0 = time.time()
        result = self.workflow.run(query)
        elapsed = time.time() - t0
        logger.info(f"VoiceAI Pipeline completed in {elapsed:.2f}s")

        return result

    def run_text(self, text: str) -> dict:
        print(f"\n [Text Mode] Input: '{text}'")
        result = self.process_query(text)
        if result:
            console_out = self.formatter.format_console(result, text)
            print(console_out)
            spoken = self.formatter.format_voice(result)
            self.tts.speak(spoken)
        return result or {}

    def run_file(self, audio_path: str) -> dict:
        print(f"\n [File Mode] Transcribing: {audio_path}")
        result_dict = self.transcriber.transcribe_file(audio_path)
        raw_text = result_dict["text"]
        print(f" Transcript: '{raw_text}'")

        result = self.process_query(raw_text)
        if result:
            console_out = self.formatter.format_console(result, raw_text)
            print(console_out)
            spoken = self.formatter.format_voice(result)
            self.tts.speak(spoken)
        return result or {}

    def run_interactive(self) -> None:
        if not self.recorder._sd_available:
            print("\n sounddevice not installed.")
            print(" Install it: pip install sounddevice")
            print(" Then re-run. Or use: python phase_7_voice_ai.py --text 'your query'")
            return

        print("\n Voice mode active. Speak your airport operations query.")
        print(" Say 'exit' or 'quit' to end.\n")
        self.tts.speak("Airport Voice AI ready. Please speak your query.")

        session_count = 0

        while True:
            try:
                print(f"\n {'-'*60}")
                print(f" Query #{session_count + 1} -- Press Ctrl+C to exit")

                audio = self.recorder.record()
                if audio is None:
                    print(" No audio captured. Try again.")
                    continue

                print(" Transcribing...")
                raw_text = self.transcriber.transcribe_array(audio)
                print(f" Heard: '{raw_text}'")

                if any(cmd in raw_text.lower() for cmd in ["exit", "quit", "stop", "bye"]):
                    print("\n Ending voice session. Goodbye!")
                    self.tts.speak("Goodbye. Airport Voice AI session ended.")
                    break

                result = self.process_query(raw_text)
                if result:
                    console_out = self.formatter.format_console(result, raw_text)
                    print(console_out)
                    spoken = self.formatter.format_voice(result)
                    self.tts.speak(spoken)
                    session_count += 1

            except KeyboardInterrupt:
                print("\n\n Session interrupted by user.")
                break
            except Exception as e:
                logger.error(f"VoiceAI Error: {e}")
                print(f" Error: {e}. Continuing...")
                continue

        print(f"\n Session complete. Queries processed: {session_count}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7 -- Airport Voice AI (Whisper + RAG + RRF + RL)"
    )
    parser.add_argument(
        "--model", type=str, default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Run in text mode (no microphone) with this query",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Transcribe an existing audio file (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable voice output (text only)",
    )
    parser.add_argument(
        "--refresh-kb", action="store_true",
        help="Rebuild the knowledge base before starting",
    )
    args = parser.parse_args()

    voice_ai = AirportVoiceAI(
        whisper_model=args.model,
        enable_tts=not args.no_tts,
        refresh_kb=args.refresh_kb,
    )

    if args.text:
        voice_ai.run_text(args.text)
    elif args.file:
        voice_ai.run_file(args.file)
    else:
        voice_ai.run_interactive()

if __name__ == "__main__":
    main()