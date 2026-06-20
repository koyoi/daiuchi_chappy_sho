#!/usr/bin/env python3
"""Convert Floodgate CSA kifu files to training data for the Transformer model.

Output format (TSV, one line per position):
  value_label  <81 board squares>  <7 black hand>  <7 white hand>  side  move_index

Usage:
  python convert_kifu.py --input kifu/floodgate --output training_data.tsv
  python convert_kifu.py --input kifu/floodgate --output training_data.tsv --min-rate 2500
  python convert_kifu.py --input kifu/floodgate --output training_data.tsv --sample-rate 0.3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# --- Piece encoding (matches C++ and nn_eval.py) ---
# 0=empty, 1-14=black pieces, 15-28=white pieces

CSA_PIECE_MAP = {
    "FU": 1, "KY": 2, "KE": 3, "GI": 4, "KI": 5, "KA": 6, "HI": 7, "OU": 8,
    "TO": 9, "NY": 10, "NK": 11, "NG": 12, "UM": 13, "RY": 14,
}

PIECE_TYPE_FROM_ID = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 1, 10: 2, 11: 3, 12: 4, 13: 6, 14: 7,  # promoted -> base type for hand
}

HAND_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}  # PieceType -> hand array index

MAX_ATTACK = 8

# --- Attack computation ---

def _step_dirs(pt: int, color: int) -> list:
    """Return step-attack directions (df, dr) in absolute coordinates."""
    fwd = -color  # Black forward = rank-1, White forward = rank+1
    if pt == 1:  # Pawn
        return [(0, fwd)]
    elif pt == 3:  # Knight
        return [(-1, 2 * fwd), (1, 2 * fwd)]
    elif pt == 4:  # Silver
        return [(-1, fwd), (0, fwd), (1, fwd), (-1, -fwd), (1, -fwd)]
    elif pt in (5, 9, 10, 11, 12):  # Gold / promoted Pawn,Lance,Knight,Silver
        return [(-1, fwd), (0, fwd), (1, fwd), (-1, 0), (1, 0), (0, -fwd)]
    elif pt == 8:  # King
        return [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    elif pt == 13:  # Horse = Bishop + adjacent orthogonal
        return [(0, -1), (0, 1), (-1, 0), (1, 0)]
    elif pt == 14:  # Dragon = Rook + adjacent diagonal
        return [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    return []


def _slide_dirs(pt: int, color: int) -> list:
    """Return slide-attack directions (df, dr) in absolute coordinates."""
    fwd = -color
    if pt == 2:  # Lance: forward only
        return [(0, fwd)]
    elif pt in (6, 13):  # Bishop / Horse
        return [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    elif pt in (7, 14):  # Rook / Dragon
        return [(0, -1), (0, 1), (-1, 0), (1, 0)]
    return []


def compute_attacks(board) -> tuple:
    """Compute per-square attack counts from current side's perspective.

    Returns (own_attacks[81], opp_attacks[81]) clamped to MAX_ATTACK.
    """
    black_atk = [0] * 81
    white_atk = [0] * 81

    for sq in range(81):
        piece = board.squares[sq]
        if piece == 0:
            continue
        color = 1 if piece > 0 else -1
        pt = abs(piece)
        f, r = file_of(sq), rank_of(sq)
        target = black_atk if color == 1 else white_atk

        for df, dr in _step_dirs(pt, color):
            nf, nr = f + df, r + dr
            if 1 <= nf <= 9 and 1 <= nr <= 9:
                target[idx(nf, nr)] += 1

        for df, dr in _slide_dirs(pt, color):
            nf, nr = f + df, r + dr
            while 1 <= nf <= 9 and 1 <= nr <= 9:
                target[idx(nf, nr)] += 1
                if board.squares[idx(nf, nr)] != 0:
                    break
                nf += df
                nr += dr

    if board.side == 1:
        own, opp = black_atk, white_atk
    else:
        own, opp = white_atk, black_atk

    return (
        [min(c, MAX_ATTACK) for c in own],
        [min(c, MAX_ATTACK) for c in opp],
    )


# Direction encoding for policy (matches nn_bridge.cpp)
DIR_UP = 0; DIR_DOWN = 1; DIR_LEFT = 2; DIR_RIGHT = 3
DIR_UL = 4; DIR_UR = 5; DIR_DL = 6; DIR_DR = 7
DIR_KNL = 8; DIR_KNR = 9


def file_of(sq: int) -> int:
    return (sq % 9) + 1

def rank_of(sq: int) -> int:
    return (sq // 9) + 1

def idx(file: int, rank: int) -> int:
    return (rank - 1) * 9 + (file - 1)


def direction_of(from_sq: int, to_sq: int, side: int) -> int:
    ff, fr = file_of(from_sq), rank_of(from_sq)
    tf, tr = file_of(to_sq), rank_of(to_sq)
    df, dr = tf - ff, tr - fr
    if side == -1:  # White
        df, dr = -df, -dr
    if df == 0 and dr < 0: return DIR_UP
    if df == 0 and dr > 0: return DIR_DOWN
    if dr == 0 and df < 0: return DIR_LEFT
    if dr == 0 and df > 0: return DIR_RIGHT
    if df < 0 and dr < 0: return DIR_UL
    if df > 0 and dr < 0: return DIR_UR
    if df < 0 and dr > 0: return DIR_DL
    if df > 0 and dr > 0: return DIR_DR
    if df == -1 and dr == -2: return DIR_KNL
    if df == 1 and dr == -2: return DIR_KNR
    return -1


def move_to_index(from_sq: int, to_sq: int, is_drop: bool, drop_piece: int,
                  promote: bool, side: int) -> int:
    if is_drop:
        base_type = PIECE_TYPE_FROM_ID.get(drop_piece, 0)
        if base_type < 1 or base_type > 7:
            return -1
        channel = 20 + (base_type - 1)
        return to_sq * 27 + channel
    d = direction_of(from_sq, to_sq, side)
    if d < 0:
        return -1
    channel = (d + 10) if promote else d
    return to_sq * 27 + channel


class Board:
    def __init__(self):
        self.squares = [0] * 81  # piece encoding
        self.black_hand = [0] * 7  # Pawn..Rook counts
        self.white_hand = [0] * 7
        self.side = 1  # 1=Black, -1=White

    def encode(self) -> List[int]:
        """Encode board state as 258 integers for NN input.

        Layout: 81 squares + 7 black hand + 7 white hand + 1 side
                + 81 own attack counts + 81 opponent attack counts.
        """
        encoded_squares = []
        for s in self.squares:
            if s == 0:
                encoded_squares.append(0)
            elif s > 0:
                encoded_squares.append(s)  # Black: 1-14
            else:
                encoded_squares.append(14 + (-s))  # White: 15-28
        base = encoded_squares + self.black_hand + self.white_hand + [0 if self.side == 1 else 1]
        own_atk, opp_atk = compute_attacks(self)
        return base + own_atk + opp_atk

    def apply_move(self, from_sq: int, to_sq: int, is_drop: bool, drop_piece: int,
                   promote: bool, side: int):
        if is_drop:
            piece = drop_piece * side
            self.squares[to_sq] = piece
            hand = self.black_hand if side == 1 else self.white_hand
            base = PIECE_TYPE_FROM_ID.get(drop_piece, 0)
            if base and HAND_INDEX.get(base) is not None:
                hi = HAND_INDEX[base]
                hand[hi] = max(0, hand[hi] - 1)
        else:
            captured = self.squares[to_sq]
            if captured != 0:
                cap_type = abs(captured)
                base = PIECE_TYPE_FROM_ID.get(cap_type, 0)
                if base and HAND_INDEX.get(base) is not None:
                    hi = HAND_INDEX[base]
                    captor_hand = self.black_hand if side == 1 else self.white_hand
                    captor_hand[hi] += 1

            piece = self.squares[from_sq]
            if promote:
                pt = abs(piece)
                if 1 <= pt <= 4:
                    promoted = pt + 8  # Pawn->ProPawn, etc.
                elif pt == 6:
                    promoted = 13  # Bishop->Horse
                elif pt == 7:
                    promoted = 14  # Rook->Dragon
                else:
                    promoted = pt
                piece = promoted * (1 if piece > 0 else -1)

            self.squares[to_sq] = piece
            self.squares[from_sq] = 0

        self.side = -self.side


def parse_csa_initial_board(lines: List[str]) -> Board:
    """Parse CSA board setup lines (P1-P9, P+, P-)."""
    board = Board()
    for line in lines:
        if re.match(r"^P[1-9]", line) and len(line) >= 29:
            rank = int(line[1])
            for cell in range(9):
                pos = 2 + cell * 3
                token = line[pos:pos + 3]
                file = 9 - cell
                sq = idx(file, rank)
                if token[0] in ('+', '-'):
                    color = 1 if token[0] == '+' else -1
                    piece_name = token[1:3]
                    pt = CSA_PIECE_MAP.get(piece_name, 0)
                    if pt:
                        board.squares[sq] = pt * color
                else:
                    board.squares[sq] = 0
        elif line.startswith("P+") or line.startswith("P-"):
            color = 1 if line[1] == '+' else -1
            hand = board.black_hand if color == 1 else board.white_hand
            for i in range(2, len(line) - 3, 4):
                piece_name = line[i + 2:i + 4]
                pt = CSA_PIECE_MAP.get(piece_name, 0)
                base = PIECE_TYPE_FROM_ID.get(pt, 0)
                if base and HAND_INDEX.get(base) is not None:
                    hand[HAND_INDEX[base]] += 1
    return board


def is_hirate(lines: List[str]) -> bool:
    """Check if position lines represent the standard starting position."""
    board_lines = [l for l in lines if re.match(r"^P[1-9]", l)]
    if len(board_lines) != 9:
        return False
    expected = [
        "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY",
        "P2 * -HI *  *  *  *  * -KA * ",
        "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU",
        "P4 *  *  *  *  *  *  *  *  * ",
        "P5 *  *  *  *  *  *  *  *  * ",
        "P6 *  *  *  *  *  *  *  *  * ",
        "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU",
        "P8 * +KA *  *  *  *  * +HI * ",
        "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY",
    ]
    for bl, ex in zip(board_lines, expected):
        if bl.strip() != ex.strip():
            return False
    return True


def parse_csa_move(line: str) -> Optional[Tuple[int, int, bool, int, bool, int]]:
    """Parse a CSA move line like '+7776FU' or '-0034FU'.

    Returns (from_sq, to_sq, is_drop, drop_piece, promote, side) or None.
    """
    if len(line) < 7:
        return None
    side_char = line[0]
    if side_char not in ('+', '-'):
        return None
    side = 1 if side_char == '+' else -1

    from_file = int(line[1])
    from_rank = int(line[2])
    to_file = int(line[3])
    to_rank = int(line[4])
    piece_name = line[5:7]
    pt = CSA_PIECE_MAP.get(piece_name, 0)
    if pt == 0:
        return None

    is_drop = (from_file == 0 and from_rank == 0)
    to_sq = idx(to_file, to_rank)

    if is_drop:
        return (-1, to_sq, True, pt, False, side)

    from_sq = idx(from_file, from_rank)
    old_piece = 0  # We don't have the board here, but we can detect promotion
    promote = (pt >= 9)  # promoted piece type means promotion happened
    # For promotion detection: if the piece name is a promoted type, it might be
    # a promotion move OR just moving an already-promoted piece.
    # We'll handle this in the game replay instead.
    return (from_sq, to_sq, False, 0, False, side)


def parse_game(filepath: Path, min_rate: int = 0) -> Optional[List[dict]]:
    """Parse a CSA kifu file and return list of position samples.

    Each sample: {board_encoded, move_index, value_label}
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None

    lines = [l.rstrip("\n\r") for l in lines]

    # Extract result
    result_line = None
    for line in lines:
        if line.startswith("'summary:"):
            result_line = line
            break
    if not result_line:
        # Check for %TORYO, %SENNICHITE, etc.
        for line in lines:
            if line == "%TORYO":
                # The side that played %TORYO loses
                pass
        return None

    # Parse result: 'summary:toryo:NAME1 lose:NAME2 win
    black_win = None
    if "lose" in result_line and "win" in result_line:
        # Format: 'summary:TYPE:PLAYER1 RESULT1:PLAYER2 RESULT2
        parts = result_line.split(":")
        if len(parts) >= 4:
            p1_result = parts[2].split()[-1] if parts[2] else ""
            p2_result = parts[3].split()[-1] if len(parts) > 3 and parts[3] else ""
            if p1_result == "win":
                black_win = True
            elif p2_result == "win":
                black_win = False
            elif p1_result == "lose":
                black_win = False
            elif p2_result == "lose":
                black_win = True
    if black_win is None:
        return None

    # Extract ratings
    if min_rate > 0:
        black_rate = 0
        white_rate = 0
        for line in lines:
            if line.startswith("'black_rate:"):
                m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
                if m:
                    black_rate = int(float(m.group(1)))
            elif line.startswith("'white_rate:"):
                m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
                if m:
                    white_rate = int(float(m.group(1)))
        if black_rate < min_rate or white_rate < min_rate:
            return None

    # Parse initial position
    board_setup_lines = [l for l in lines if re.match(r"^P[1-9+\-]", l)]
    if is_hirate(board_setup_lines):
        board = Board()
        # Set up standard position
        std_lines = [
            "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY",
            "P2 * -HI *  *  *  *  * -KA * ",
            "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU",
            "P4 *  *  *  *  *  *  *  *  * ",
            "P5 *  *  *  *  *  *  *  *  * ",
            "P6 *  *  *  *  *  *  *  *  * ",
            "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU",
            "P8 * +KA *  *  *  *  * +HI * ",
            "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY",
        ]
        board = parse_csa_initial_board(std_lines)
    else:
        board = parse_csa_initial_board(board_setup_lines)

    # Find initial side
    for line in lines:
        if line == "+":
            board.side = 1
            break
        elif line == "-":
            board.side = -1
            break

    # Parse moves
    move_lines = []
    for line in lines:
        if len(line) >= 7 and line[0] in ('+', '-') and line[1].isdigit():
            move_lines.append(line)

    if len(move_lines) < 10:
        return None

    samples = []
    total_moves = len(move_lines)

    for ply, mline in enumerate(move_lines):
        if len(mline) < 7:
            continue
        side_char = mline[0]
        side = 1 if side_char == '+' else -1

        from_file = int(mline[1])
        from_rank = int(mline[2])
        to_file = int(mline[3])
        to_rank = int(mline[4])
        piece_name = mline[5:7]
        pt = CSA_PIECE_MAP.get(piece_name, 0)
        if pt == 0:
            continue

        is_drop = (from_file == 0 and from_rank == 0)
        to_sq = idx(to_file, to_rank)

        if is_drop:
            from_sq = -1
            promote = False
            drop_piece = pt
        else:
            from_sq = idx(from_file, from_rank)
            drop_piece = 0
            # Detect promotion: CSA gives the resulting piece type
            # If the piece on from_sq is a base type and the move's piece is promoted, it's promotion
            current_piece = abs(board.squares[from_sq]) if from_sq >= 0 else 0
            promote = (pt >= 9 and current_piece < 9) if current_piece > 0 else False

        # Compute move index
        midx = move_to_index(from_sq, to_sq, is_drop, drop_piece if is_drop else 0,
                             promote, side)

        # Value label: from the perspective of the moving side
        if side == 1:
            value = 1.0 if black_win else -1.0
        else:
            value = -1.0 if black_win else 1.0

        if midx >= 0:
            encoded = board.encode()
            samples.append({
                "encoded": encoded,
                "move_index": midx,
                "value": value,
                "ply": ply,
                "total": total_moves,
            })

        # Apply move to board
        board.apply_move(from_sq, to_sq, is_drop, drop_piece if is_drop else pt,
                         promote, side)

    return samples


def sample_positions(samples: List[dict], sample_rate: float = 1.0,
                     opening_n: int = 20, endgame_n: int = 20) -> List[dict]:
    """Sample positions from a game.

    Always include first opening_n and last endgame_n moves.
    Sample middle moves at sample_rate.
    """
    import random

    if not samples:
        return []

    total = len(samples)
    if total <= opening_n + endgame_n:
        return samples

    result = []
    for s in samples:
        ply = s["ply"]
        if ply < opening_n or ply >= total - endgame_n:
            result.append(s)
        elif random.random() < sample_rate:
            result.append(s)

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert Floodgate CSA kifu to training data")
    parser.add_argument("--input", required=True, help="Root directory containing CSA files")
    parser.add_argument("--output", required=True, help="Output TSV file path")
    parser.add_argument("--min-rate", type=int, default=2500,
                        help="Minimum rating for both players (default: 2500)")
    parser.add_argument("--sample-rate", type=float, default=0.3,
                        help="Sampling rate for middle-game positions (0.0-1.0)")
    parser.add_argument("--opening-n", type=int, default=20,
                        help="Always include first N moves")
    parser.add_argument("--endgame-n", type=int, default=20,
                        help="Always include last N moves before end")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Max games to process (0=all)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    csa_files = sorted(input_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)

    if args.max_games > 0:
        csa_files = csa_files[:args.max_games]

    total_samples = 0
    total_games = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for i, filepath in enumerate(csa_files):
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(csa_files)} games, {total_samples} samples...",
                      file=sys.stderr)

            samples = parse_game(filepath, min_rate=args.min_rate)
            if samples is None:
                skipped += 1
                continue

            sampled = sample_positions(samples, args.sample_rate,
                                       args.opening_n, args.endgame_n)

            for s in sampled:
                parts = [f"{s['value']:.1f}"]
                parts.extend(str(v) for v in s["encoded"])
                parts.append(str(s["move_index"]))
                out.write("\t".join(parts) + "\n")
                total_samples += 1

            total_games += 1

    print(f"Done: {total_games} games, {total_samples} samples "
          f"(skipped {skipped} games)", file=sys.stderr)


if __name__ == "__main__":
    main()
