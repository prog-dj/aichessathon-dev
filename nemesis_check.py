"""Run SF over every nemesis game: eval trajectory for OUR side + our biggest blunders."""
import glob, os, sys
import chess, chess.pgn, chess.engine

SF = r"C:\Users\44783\AI_Chessathon\sp-gen\stockfish\stockfish-windows-x86-64-universal.exe"
PGN_DIR = r"C:\Users\44783\AI_Chessathon\aichessathon-dev\analysis\pgns\nemeses"
US = "ChessNotCheckers"
DEPTH = 14

eng = chess.engine.SimpleEngine.popen_uci(SF)
eng.configure({"Threads": 4, "Hash": 512})


def cp(info):
    s = info["score"].white().score(mate_score=100000)
    return s


for path in sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn"))):
    g = chess.pgn.read_game(open(path))
    white = g.headers.get("White", "?")
    we_white = white == US
    opp = g.headers.get("Black") if we_white else white
    board = g.board()
    evals = [cp(eng.analyse(board, chess.engine.Limit(depth=DEPTH)))]
    moves = list(g.mainline_moves())
    san_list = []
    tmp = g.board()
    for mv in moves:
        san_list.append(tmp.san(mv)); tmp.push(mv)
    blunders = []
    for i, mv in enumerate(moves):
        mover_is_us = (board.turn == chess.WHITE) == we_white
        board.push(mv)
        e = cp(eng.analyse(board, chess.engine.Limit(depth=DEPTH)))
        evals.append(e)
        if mover_is_us:
            # eval from our POV before vs after
            before = evals[i] if we_white else -evals[i]
            after = e if we_white else -e
            drop = before - after
            if drop >= 120:
                blunders.append((i // 2 + 1, san_list[i], before, after, drop))
    # our-POV eval trajectory sampled
    ours = [(e if we_white else -e) for e in evals]
    print("=" * 90)
    print(f"{os.path.basename(path)}  |  we are {'White' if we_white else 'Black'} vs {opp}  |  result {g.headers.get('Result')}")
    mv0 = g.board().fullmove_number
    marks = [(mv0 + k * 3, ours[k * 6]) for k in range(len(ours) // 6)]
    print("  our eval:  " + "  ".join(f"m{m}:{v/100:+.1f}" for m, v in marks))
    if blunders:
        print("  our moves that dropped >=1.2:")
        for mno, san, b, a, d in blunders:
            print(f"    move {mno} {san:7s}  {b/100:+.1f} -> {a/100:+.1f}   (lost {d/100:.1f})")
    else:
        print("  no single move dropped >=1.2 -- lost gradually or was already lost")

eng.quit()
