from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_airport_dataset
from phase_6_agent_workflow import AirportAgenticWorkflow

def default_query() -> str:
    df  = load_airport_dataset()
    row = df.sort_values("departure_delay_minutes", ascending=False).iloc[0]
    return (
        f"Why is flight {row['flight_id']} operationally high risk, "
        "what dependencies are causing the disruption, and what should the airport do next?"
    )

def _print_standard_result(result: dict) -> None:
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Flight selected   : {result.get('selected_flight_id', '—')}")
    print(f"  Operation ID      : {result.get('selected_operation_id', '—')}")

    reco = result.get("rl_recommendation", {})
    print(f"\n  RL Recommendation :")
    print(f"    Action            : {reco.get('action', '—')}")
    print(f"    Policy mode       : {reco.get('policy_mode', '—')}")
    print(f"    Expected reduction: {reco.get('expected_delay_reduction_minutes', 0)} min")
    print(f"    Projected delay   : {reco.get('projected_delay_minutes', '—')} min")
    print(f"    Reason            : {reco.get('reason', '—')}")

    print("\n  Top Fused Evidence (RAG + RRF):")
    for item in result.get("fused_hits", [])[:5]:
        print(f"    {item.rank}. [{item.retriever}] {item.doc.title}  (score={item.score:.4f})")

    print("\n" + "=" * 65)
    print("  FINAL ANSWER")
    print("=" * 65)
    print(result.get("final_answer", "No answer produced."))

def _pick(prompt: str, options: list[str]) -> str:
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = input("\n  Enter choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            chosen = options[int(raw) - 1]
            print(f"  -> {chosen}\n")
            return chosen
        print(f"  Please enter a number between 1 and {len(options)}.")

def _pick_index(prompt: str, options: list[str]) -> int:
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = input("\n  Enter choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            idx = int(raw) - 1
            print(f"  -> {options[idx]}\n")
            return idx
        print(f"  Please enter a number between 1 and {len(options)}.")

def interactive_menu(workflow: AirportAgenticWorkflow) -> None:
    print("\n" + "=" * 65)
    print("  Smart Airport Agentic AI — Interactive Console")
    print("=" * 65)

    while True:
        mode = _pick(
            "Select mode:",
            ["Text Query", "Voice Query (Phase 7)", "Exit"],
        )

        if mode == "Exit":
            print("  Goodbye!\n")
            break

        elif mode == "Text Query":
            print("  Type your airport operations query below.")
            print("  (Press Enter without typing to use the default query.)\n")
            raw = input("  Query: ").strip()
            query = raw if raw else default_query()
            if not raw:
                print(f"  Using default query:\n  {query}")

            print("\n  ⚙  Running pipeline...")
            result = workflow.run(query)
            _print_standard_result(result)

        elif mode == "Voice Query (Phase 7)":
            _run_voice_interactive(workflow)

def _run_voice_interactive(workflow: AirportAgenticWorkflow) -> None:
    try:
        from Phase_7_voice_ai import AirportVoiceAI
    except ImportError as exc:
        print(f"\n  ❌  Cannot import Phase 7: {exc}")
        print("  Install required packages:")
        print("      pip install openai-whisper gtts sounddevice playsound")
        return

    tts_choice = _pick(
        "Enable voice output (text-to-speech)?",
        ["Yes — speak the answer aloud", "No — text only"],
    )
    enable_tts = tts_choice.startswith("Yes")

    print("\n  [Phase 7] Initialising Voice AI pipeline...")
    print("  (Whisper small + RAG + RRF + Graph Intelligence + RL)\n")

    voice_ai = AirportVoiceAI(
        whisper_model = "small",
        enable_tts    = enable_tts,
        refresh_kb    = False,
    )
    voice_ai.workflow = workflow

    VOICE_OPTS = [
        "Live Microphone  — speak your query now",
        "Type a query     — run it through the voice pipeline",
        "Audio File       — transcribe a WAV / MP3 file",
    ]
    sub_idx = _pick_index("Choose voice input method:", VOICE_OPTS)

    if sub_idx == 0:
        if not voice_ai.recorder._sd_available:
            print("\n  sounddevice is not installed — cannot use live microphone.")
            print("  Install it:  pip install sounddevice")
            return

        print("\n  Microphone mode active. Speak your airport operations query.")
        print("  Say 'exit' or 'quit' to return to the main menu.\n")

        audio = voice_ai.recorder.record()
        if audio is None:
            print("  No speech detected. Returning to menu.")
            return

        print("  Transcribing...")
        raw_text = voice_ai.transcriber.transcribe_array(audio)
        print(f"  You said: '{raw_text}'\n")

        if any(cmd in raw_text.lower() for cmd in ["exit", "quit", "stop"]):
            print("  Returning to main menu.")
            return

        result = voice_ai.process_query(raw_text)
        if result:
            console_out = voice_ai.formatter.format_console(result, raw_text)
            print(console_out)
            if enable_tts:
                voice_ai.tts.speak(voice_ai.formatter.format_voice(result))
            _print_standard_result(result)

    elif sub_idx == 1:
        print("  Type your query and press Enter.\n")
        raw = input("  Query: ").strip()
        if not raw:
            print("  No input. Returning to menu.")
            return

        result = voice_ai.run_text(raw)
        if result:
            _print_standard_result(result)

    elif sub_idx == 2:
        print("  Enter the path to your audio file (WAV, MP3, M4A, etc.):")
        file_path = input("  File path: ").strip().strip('"').strip("'")
        if not Path(file_path).exists():
            print(f"  File not found: {file_path}")
            return

        result = voice_ai.run_file(file_path)
        if result:
            _print_standard_result(result)

def run_standard_cli(args, workflow: AirportAgenticWorkflow) -> None:
    query = args.query or default_query()
    print(f"\n  Running query:\n  {query}\n")
    result = workflow.run(query)
    _print_standard_result(result)

def run_voice_cli(args, workflow: AirportAgenticWorkflow) -> None:
    try:
        from Phase_7_voice_ai import AirportVoiceAI
    except ImportError as exc:
        print(f"\n  ❌  Cannot import Phase 7: {exc}")
        print("  pip install openai-whisper gtts sounddevice playsound")
        sys.exit(1)

    voice_ai = AirportVoiceAI(
        whisper_model = args.whisper_model,
        enable_tts    = not args.no_tts,
        refresh_kb    = False,
    )
    voice_ai.workflow = workflow

    if args.voice_text:
        result = voice_ai.run_text(args.voice_text)
        if result:
            _print_standard_result(result)
    elif args.voice_file:
        if not Path(args.voice_file).exists():
            print(f"  ❌  File not found: {args.voice_file}")
            sys.exit(1)
        result = voice_ai.run_file(args.voice_file)
        if result:
            _print_standard_result(result)
    else:
        if not voice_ai.recorder._sd_available:
            print("  ⚠️  sounddevice not installed.  pip install sounddevice")
            sys.exit(1)
        voice_ai.run_interactive()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Airport Agentic AI — run without flags for interactive menu.",
    )

    parser.add_argument("--query",       type=str,  default=None,
                        help="Text query (skips interactive menu).")
    parser.add_argument("--refresh-kb",  action="store_true",
                        help="Rebuild the knowledge base before running.")
    parser.add_argument("--voice",       action="store_true",
                        help="Live microphone mode (skips interactive menu).")
    parser.add_argument("--voice-text",  type=str,  default=None, metavar="QUERY",
                        help="Voice pipeline with typed input (no mic).")
    parser.add_argument("--voice-file",  type=str,  default=None, metavar="PATH",
                        help="Voice pipeline with audio file input.")
    parser.add_argument("--whisper-model",type=str,  default="small",
                        choices=["tiny","base","small","medium","large"],
                        help="Whisper model size (default: small).")
    parser.add_argument("--no-tts",      action="store_true",
                        help="Disable text-to-speech output.")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  Smart Airport Agentic AI — Initialising pipeline...")
    print("=" * 65)
    print("  Building workflow (KB + Retrievers + Graph + RL)...")
    workflow = AirportAgenticWorkflow(refresh_knowledge=args.refresh_kb)
    print("  Workflow ready.\n")

    cli_flags_given = any([
        args.query, args.voice, args.voice_text, args.voice_file
    ])

    if not cli_flags_given:
        interactive_menu(workflow)

    elif args.voice or args.voice_text or args.voice_file:
        run_voice_cli(args, workflow)

    else:
        run_standard_cli(args, workflow)

if __name__ == "__main__":
    main()