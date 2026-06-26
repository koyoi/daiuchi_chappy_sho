"""ShogiGUI simulator: measures wall time per move, detects time violations."""
import subprocess
import sys
import time

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "build/Release/kishi-to-nnue.exe"
MAIN_TIME_MS = 60_000
BYOYOMI_MS = 10_000
MAX_MOVE_TIME_MS = 0  # 0 = use engine default
INITIAL_MOVES = "2g2f 3c3d"


def log(msg):
    t = time.strftime("%H:%M:%S")
    ms = int(time.time() * 1000) % 1000
    print(f"[{t}.{ms:03d}] {msg}", flush=True)


def run():
    proc = subprocess.Popen(
        [ENGINE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )

    def send(cmd):
        log(f">>> {cmd}")
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except (OSError, BrokenPipeError):
            log("!!! PIPE BROKEN (engine crashed?)")
            return False
        return True

    def read_until(prefix, timeout=30):
        lines = []
        deadline_t = time.time() + timeout
        while time.time() < deadline_t:
            try:
                line = proc.stdout.readline()
            except Exception:
                log("!!! STDOUT READ ERROR")
                break
            if not line:
                log("!!! ENGINE EOF (process terminated)")
                break
            line = line.strip()
            if line:
                lines.append(line)
                log(f"<<< {line}")
            if line.startswith(prefix):
                return lines
        if not any(l.startswith(prefix) for l in lines):
            log(f"!!! TIMEOUT waiting for '{prefix}' ({timeout:.1f}s)")
        return lines

    # USI handshake
    send("usi")
    read_until("usiok")
    send("setoption name Book value false")
    if MAX_MOVE_TIME_MS > 0:
        send(f"setoption name MaxMoveTimeMs value {MAX_MOVE_TIME_MS}")
    send("isready")
    read_until("readyok")
    send("usinewgame")

    btime = MAIN_TIME_MS
    wtime = MAIN_TIME_MS

    all_moves = INITIAL_MOVES.strip().split()
    is_black = (len(all_moves) % 2 == 0)

    violations = []
    move_count = 0
    max_total_moves = 80

    for _ in range(max_total_moves):
        move_count += 1
        side_name = "BLACK" if is_black else "WHITE"
        remaining = btime if is_black else wtime
        hard_limit = BYOYOMI_MS if remaining <= 0 else remaining + BYOYOMI_MS

        position_cmd = "position startpos moves " + " ".join(all_moves)
        log(f"--- Move {move_count} ({side_name}) btime={btime} wtime={wtime} byoyomi={BYOYOMI_MS} ---")

        send(position_cmd)
        go_cmd = f"go btime {btime} wtime {wtime} byoyomi {BYOYOMI_MS}"
        t0 = time.time()
        if not send(go_cmd):
            log("!!! ENGINE DEAD, stopping")
            break

        read_timeout = min(hard_limit / 1000 + 5, 120)
        lines = read_until("bestmove", timeout=read_timeout)
        t1 = time.time()
        elapsed_ms = int((t1 - t0) * 1000)

        bestmove_line = None
        for l in lines:
            if l.startswith("bestmove"):
                bestmove_line = l
                break

        if bestmove_line is None:
            log(f"!!! NO BESTMOVE after {elapsed_ms}ms - ENGINE HUNG/CRASHED")
            violations.append((move_count, side_name, elapsed_ms, "NO_BESTMOVE"))
            break

        move_str = bestmove_line.split()[1] if len(bestmove_line.split()) > 1 else "resign"
        if move_str == "resign":
            log(f"Engine resigned at move {move_count}")
            break

        # Check time violation (byoyomi hard limit when no remaining time)
        if remaining <= 0 and elapsed_ms > BYOYOMI_MS:
            log(f"!!! BYOYOMI VIOLATION: {elapsed_ms}ms > {BYOYOMI_MS}ms")
            violations.append((move_count, side_name, elapsed_ms, "BYOYOMI_VIOLATION"))
        elif elapsed_ms > hard_limit:
            log(f"!!! TIME VIOLATION: {elapsed_ms}ms > {hard_limit}ms")
            violations.append((move_count, side_name, elapsed_ms, "TOTAL_TIME_VIOLATION"))

        log(f"Move {move_count}: {move_str} in {elapsed_ms}ms (remaining={remaining}, limit={hard_limit}ms)")

        if is_black:
            btime = max(0, btime - elapsed_ms)
        else:
            wtime = max(0, wtime - elapsed_ms)

        all_moves.append(move_str)
        is_black = not is_black

    # Cleanup
    try:
        send("quit")
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if violations:
        print(f"TIME VIOLATIONS: {len(violations)}")
        for v in violations:
            print(f"  Move {v[0]} ({v[1]}): {v[2]}ms - {v[3]}")
    else:
        print("No time violations detected.")
    print(f"Total moves played: {move_count}")
    print(f"Final times: btime={btime}ms wtime={wtime}ms")


if __name__ == "__main__":
    run()
