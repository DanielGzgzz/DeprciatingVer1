import time
import os
import sys
import math
import random
from datetime import datetime

# ANSI Color Codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
RESET = '\033[0m'
CLEAR = '\033[2J'
HOME = '\033[H'
BG_RED = '\033[41;97m'
BG_GREEN = '\033[42;30m'
BLINK = '\033[5m'

# Simulated telemetry values that would come from market_analyzer.py in a live hook
def get_telemetry(cycle):
    # Simulate a sudden market shock at cycle 15
    fiedler_lambda2 = max(0.1, 1.2 - (cycle / 30.0)) if cycle < 15 else max(0.01, 0.5 - ((cycle - 15) / 5.0))
    r_t = min(0.99, 0.3 + (cycle / 40.0)) if cycle < 15 else min(0.99, 0.7 + ((cycle - 15) / 10.0))
    r_dot_t = 0.01 if cycle < 15 else 0.15 # Acceleration spike

    interlock = "SAFE" if (r_t > 0.8 or r_dot_t > 0.1) else "ARMED"
    return fiedler_lambda2, r_t, r_dot_t, interlock

def draw_scatter_hud(cycle, width=40, height=12):
    # Vector Trajectory HUD (Phase Space Mapping)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    center_x = width // 2
    center_y = height // 2

    # As cycle increases, cluster tighter to simulate synchronization
    scatter_radius = max(1, 15 - (cycle // 2))

    for _ in range(12): # Plot 12 core assets
        # Random position within decreasing radius
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, scatter_radius)
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle) * (height/width)) # scale for char aspect ratio

        # Keep inside bounds
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        grid[y][x] = "*"

    # Draw crosshairs
    for i in range(width):
        if grid[center_y][i] == " ": grid[center_y][i] = "-"
    for i in range(height):
        if grid[i][center_x] == " ": grid[i][center_x] = "|"
    grid[center_y][center_x] = "+"

    return ["".join(line) for line in grid]

def print_dashboard(cycle, threat_log):
    sys.stdout.write(CLEAR + HOME)
    fiedler, r_t, r_dot_t, interlock = get_telemetry(cycle)

    # 1. Subsystem BIT Matrix & Interlock
    interlock_str = f"{BG_GREEN}{BOLD} [INTERLOCK STATUS: ARMED] {RESET}" if interlock == "ARMED" else f"{BG_RED}{BLINK}{BOLD} [INTERLOCK STATUS: DE-ARMED / SAFE] {RESET}"

    print(f"┌────────────────────────────────────────────────────────────────────────┐")
    print(f"│  [BIT] DATA: {GREEN}OK{RESET} | COV: {GREEN}OK{RESET} | SOLVER: {GREEN}OK{RESET}    {interlock_str:>45}│")
    print(f"├──────────────────────────────────────┬─────────────────────────────────┤")

    # 2. Portfolio Allocation vs Vector HUD
    hud_lines = draw_scatter_hud(cycle)
    allocations = [
        f"  XOM: {CYAN}████████████{RESET} 32%               ",
        f"  CVX: {CYAN}████████{RESET} 21%                   ",
        f"  TLT: {MAGENTA}{'████████████████████' if interlock == 'SAFE' else '██'}{RESET} {'54%' if interlock == 'SAFE' else '4%'}       ",
        f"  TSM: {CYAN}█{RESET} 2%                            ",
        "                                       ",
        f"  {YELLOW}Fiedler (λ2):{RESET} {fiedler:.3f}                ",
        f"  {YELLOW}Sync Mag (r):{RESET} {r_t:.3f}                ",
        f"  {YELLOW}Phase Accel (r_dot):{RESET} {r_dot_t:+.3f}        ",
        "                                       ",
        "                                       ",
        "                                       ",
        "                                       "
    ]

    for i in range(len(hud_lines)):
        # Pad allocation string to fixed width for clean table
        left_pad = f"{allocations[i]:<38}"
        print(f"│{left_pad}│ {MAGENTA}{hud_lines[i]}{RESET} │")

    print(f"├──────────────────────────────────────┴─────────────────────────────────┤")

    # 3. Sequential Threat Matrix
    print(f"│  {CYAN}{BOLD}[SEQUENTIAL THREAT MATRIX]{RESET}                                            │")

    # Generate threats based on telemetry
    now = datetime.now().strftime('%H:%M:%S')
    if cycle == 1:
        threat_log.append(f"{now} | SYS_BIT  | LAPLACIAN MATRIX INVERSION SUCCESSFUL")
    elif cycle == 10:
        threat_log.append(f"{now} | TELEM    | FIEDLER DECAY DETECTED: LAMBDA_2 < 0.85")
    elif cycle == 15:
        threat_log.append(f"{now} | WARN     | PHASE COHERENCE ACCELERATING: r_dot = +0.15/sec")
        threat_log.append(f"{now} | CRITICAL | {RED}SYSTEM INTERLOCK TRIGGERED{RESET}")
    elif cycle == 16:
        threat_log.append(f"{now} | ACTION   | {MAGENTA}CAPITAL EVACUATION INITIATED -> ROTATING TO HAVENS{RESET}")

    # Print last 4 logs
    for log in threat_log[-4:]:
        # Strip ansi codes to calculate true length for right padding
        clean_len = len(log.replace(GREEN,'').replace(YELLOW,'').replace(RED,'').replace(CYAN,'').replace(MAGENTA,'').replace(BOLD,'').replace(RESET,'').replace(BLINK,'').replace(BG_RED,'').replace(BG_GREEN,''))
        pad = max(0, 70 - clean_len)
        print(f"│  {log}{' ' * pad} │")

    # Fill empty lines if log < 4
    for _ in range(4 - len(threat_log[-4:])):
        print(f"│                                                                        │")

    print(f"└────────────────────────────────────────────────────────────────────────┘")
    print("\nPress Ctrl+C to exit dashboard.")
    sys.stdout.flush()

def run_tui():
    cycle = 0
    threat_log = []
    try:
        while True:
            print_dashboard(cycle, threat_log)
            cycle += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{CYAN}Shutting down Aerospace TUI Dashboard...{RESET}")

if __name__ == "__main__":
    run_tui()
