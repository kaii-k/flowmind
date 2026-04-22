"""
flowmind_progress.py  –  Drop this file next to main.py and run it INSTEAD.
It wraps main.py in a live terminal animation so you can see FlowMind is working.

Usage (same flags as main.py):
  python flowmind_progress.py --kaggle-dataset chethuhn/network-intrusion-dataset ...
  python flowmind_progress.py --demo
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import os
import itertools

# ── ANSI colours (work in PowerShell 7+ and Windows Terminal) ─────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[38;5;84m"
CYAN   = "\033[38;5;51m"
YELLOW = "\033[38;5;220m"
RED    = "\033[38;5;203m"
GREY   = "\033[38;5;244m"
TEAL   = "\033[38;5;43m"

# Enable ANSI on older Windows consoles
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


# ── Banner ─────────────────────────────────────────────────────────────────────
BANNER = f"""
{TEAL}{BOLD}
  ███████╗██╗      ██████╗ ██╗    ██╗███╗   ███╗██╗███╗   ██╗██████╗
  ██╔════╝██║     ██╔═══██╗██║    ██║████╗ ████║██║████╗  ██║██╔══██╗
  █████╗  ██║     ██║   ██║██║ █╗ ██║██╔████╔██║██║██╔██╗ ██║██║  ██║
  ██╔══╝  ██║     ██║   ██║██║███╗██║██║╚██╔╝██║██║██║╚██╗██║██║  ██║
  ██║     ███████╗╚██████╔╝╚███╔███╔╝██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
{RESET}{GREY}  Context-Aware Flow Embeddings · Adaptive Classification · Threat Detection{RESET}
"""

# ── Spinner frames ─────────────────────────────────────────────────────────────
SPINNER_FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
PULSE_FRAMES   = ["▱▱▱▱▱","▰▱▱▱▱","▰▰▱▱▱","▰▰▰▱▱","▰▰▰▰▱","▰▰▰▰▰","▱▰▰▰▰","▱▱▰▰▰","▱▱▱▰▰","▱▱▱▱▰"]
RADAR_FRAMES   = ["◜ ","◝ "," ◞"," ◟"]
DOTS           = ["   ", ".  ", ".. ", "..."]

PHASES = [
    (GREEN,  "Downloading & caching Kaggle datasets"),
    (CYAN,   "Loading and normalising flow CSV files"),
    (YELLOW, "Building context-aware embedding vectors"),
    (TEAL,   "Fitting adaptive anomaly baseline"),
    (GREEN,  "Scoring flows · adaptive learning loop"),
    (CYAN,   "Running multi-class traffic classification"),
    (YELLOW, "Generating HTML threat analysis report"),
]


class LiveSpinner:
    """Thread that redraws a single status line while main.py runs."""

    def __init__(self) -> None:
        self._stop  = threading.Event()
        self._lines: list[str] = []
        self._lock  = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def add_line(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        spin   = itertools.cycle(SPINNER_FRAMES)
        pulse  = itertools.cycle(PULSE_FRAMES)
        phase_cycle = itertools.cycle(PHASES)
        phase_color, phase_label = next(phase_cycle)
        phase_tick = 0
        phase_duration = 60   # frames before auto-advancing phase display

        start_time = time.time()

        while not self._stop.is_set():
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            timer_str = f"{mins:02d}:{secs:02d}"

            s = next(spin)
            p = next(pulse)

            # Auto-cycle the displayed phase label every N frames
            phase_tick += 1
            if phase_tick >= phase_duration:
                phase_tick = 0
                phase_color, phase_label = next(phase_cycle)

            # Build status line
            status = (
                f"\r  {phase_color}{BOLD}{s}{RESET}  "
                f"{phase_color}{phase_label}{RESET}"
                f"{GREY} {DOTS[phase_tick % 4]}{RESET}"
                f"   {DIM}{p}{RESET}"
                f"   {GREY}elapsed {timer_str}{RESET}"
                "          "   # trailing spaces to clear old chars
            )
            sys.stdout.write(status)
            sys.stdout.flush()
            time.sleep(0.10)

        # Clear spinner line
        sys.stdout.write("\r" + " " * 90 + "\r")
        sys.stdout.flush()


def _intercept_output(proc: subprocess.Popen, spinner: LiveSpinner) -> list[str]:
    """Read stdout line-by-line, forward [FlowMind] lines to terminal, buffer all."""
    captured: list[str] = []
    assert proc.stdout is not None

    for raw in proc.stdout:
        line = raw.rstrip()
        captured.append(line)

        lo = line.lower()

        # Suppress raw Python tracebacks from the animation layer –
        # they'll be printed in full after the spinner stops.
        if line.startswith("  File ") or line.startswith("    "):
            continue

        if "[flowmind]" in lo:
            sys.stdout.write("\r" + " " * 90 + "\r")   # clear spinner
            if "warmup" in lo:
                print(f"  {CYAN}▶  {line.strip()}{RESET}")
            elif "processed batch" in lo:
                print(f"  {YELLOW}↻  {line.strip()}{RESET}")
            elif "training source" in lo:
                print(f"  {GREEN}◈  {line.strip()}{RESET}")
            elif "capped" in lo or "generated" in lo or "loaded" in lo:
                print(f"  {TEAL}◆  {line.strip()}{RESET}")
            else:
                print(f"  {GREY}·  {line.strip()}{RESET}")

        elif line.startswith("Downloaded") or line.startswith("Loaded") or line.startswith("Using"):
            sys.stdout.write("\r" + " " * 90 + "\r")
            print(f"  {TEAL}◆  {line.strip()}{RESET}")

    return captured


def _print_final_summary(lines: list[str]) -> None:
    """Pretty-print the final FlowMind AI summary from captured output."""
    in_summary   = False
    in_top_flows = False
    in_breakdown = False
    indent_next  = False

    print(f"\n{TEAL}{BOLD}{'─'*64}{RESET}")
    print(f"{TEAL}{BOLD}  FlowMind AI — Results{RESET}")
    print(f"{TEAL}{BOLD}{'─'*64}{RESET}\n")

    for line in lines:
        stripped = line.strip()

        if stripped == "FlowMind AI summary":
            in_summary   = True
            in_top_flows = False
            continue

        if stripped == "Top suspicious flows":
            in_summary   = False
            in_top_flows = True
            print(f"\n  {CYAN}{BOLD}Top Suspicious Flows{RESET}")
            print(f"  {GREY}{'─'*60}{RESET}")
            continue

        if in_summary:
            if stripped.startswith("- class_breakdown:"):
                in_breakdown = True
                print(f"  {CYAN}{'Traffic Classes':<30}{RESET}")
                continue
            if in_breakdown:
                if stripped.startswith("- "):
                    in_breakdown = False
                else:
                    cls_part = stripped.strip()
                    if ":" in cls_part:
                        name, count = cls_part.split(":", 1)
                        name  = name.strip()
                        count = count.strip()
                        color = _class_color(name)
                        print(f"    {color}{name:<22}{RESET} {BOLD}{count}{RESET}")
                    continue

            if stripped.startswith("- "):
                kv = stripped[2:]
                if ": " in kv:
                    key, val = kv.split(": ", 1)
                    key_clean = key.replace("_", " ").title()
                    color = _metric_color(key, val)
                    print(f"  {GREY}{key_clean:<30}{RESET} {color}{BOLD}{val}{RESET}")
                else:
                    print(f"  {GREY}{kv}{RESET}")

        if in_top_flows and stripped.startswith("- "):
            flow = stripped[2:]
            # Colour code by risk
            if "risk=critical" in flow:
                c = RED
            elif "risk=high" in flow:
                c = YELLOW
            elif "risk=medium" in flow:
                c = CYAN
            else:
                c = GREEN

            # Highlight class= token
            parts = flow.split(" class=")
            if len(parts) == 2:
                cls_name = parts[1].split(" ")[0]
                cls_c    = _class_color(cls_name)
                flow = parts[0] + f" class={cls_c}{BOLD}{cls_name}{RESET}{c}" + parts[1][len(cls_name):]

            print(f"  {c}►  {flow}{RESET}")

    print(f"\n{TEAL}{BOLD}{'─'*64}{RESET}\n")


def _class_color(name: str) -> str:
    return {
        "BENIGN":          GREEN,
        "DoS/DDoS":        RED,
        "Port Scan":       YELLOW,
        "Brute Force":     RED,
        "Web Attack":      YELLOW,
        "Botnet":          RED,
        "Infiltration":    YELLOW,
        "Unknown Anomaly": GREY,
    }.get(name, CYAN)


def _metric_color(key: str, val: str) -> str:
    if "recall" in key or "detected" in key:
        try:
            return GREEN if float(val.rstrip("%")) > 50 else YELLOW
        except ValueError:
            pass
    if "false_positive" in key:
        try:
            return GREEN if float(val.rstrip("%")) < 5 else RED
        except ValueError:
            pass
    if "flag_rate" in key:
        try:
            return YELLOW if float(val.rstrip("%")) > 50 else GREEN
        except ValueError:
            pass
    return RESET


def main() -> None:
    # Forward all sys.argv args except our own script name to main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target     = os.path.join(script_dir, "main.py")
    cmd        = [sys.executable, target] + sys.argv[1:]

    print(BANNER)
    print(f"  {GREY}Command: {' '.join(sys.argv[1:]) or '--demo'}{RESET}\n")

    spinner = LiveSpinner()
    spinner.start()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured = _intercept_output(proc, spinner)
        proc.wait()
    finally:
        spinner.stop()

    # Check for errors
    exit_code = proc.returncode
    if exit_code != 0:
        print(f"\n  {RED}{BOLD}✗ FlowMind exited with error (code {exit_code}){RESET}\n")
        # Print any error lines
        for line in captured:
            if "error" in line.lower() or "traceback" in line.lower() or line.startswith("  File"):
                print(f"  {RED}{line}{RESET}")
        sys.exit(exit_code)

    # Success — print formatted summary
    _print_final_summary(captured)

    # Find and print report path
    for line in reversed(captured):
        if "report:" in line.lower() and ".html" in line:
            path = line.split("report:")[-1].strip()
            print(f"  {GREEN}{BOLD}✓ Report ready →{RESET} {BOLD}{path}{RESET}\n")
            break


if __name__ == "__main__":
    main()