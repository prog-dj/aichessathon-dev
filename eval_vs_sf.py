"""Our engine's eval vs Stockfish's, position by position, over the nemesis games.
Shows where our judgment diverges from reality - i.e. what SF labels would fix."""
import glob, os
os.environ["FASTCHESS_NNUE"] = "1"
import chess, chess.pgn, chess.engine
import fastchess as fc

SF = r"C:\Users\44783\AI_Chessathon\sp-gen\stockfish\stockfish-windows-x86-64-universal.exe"
PGN_DIR = r"C:\Users\44783\AI_Chessathon\aichessathon-dev\analysis\pgns\nemeses"
US = "ChessNotCheckers"
OUR_NODES = 25000
SF_DEPTH = 14

sf = chess.engine.SimpleEngine.popen_uci(SF)
sf.configure({"Threads": 3, "Hash": 256})
eng = fc.Engine()

alldiffs = []
for path in sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn"))):
    g = chess.pgn.read_game(open(path))
    we_white = g.headers.get("White") == US
    board = g.board()
    moves = list(g.mainline_moves())
    ucis = []
    rows = []
    for i, mv in enumerate(moves):
        # eval the position *before* each of our moves + every ~4th position
        our_turn = (board.turn == chess.WHITE) == we_white
        if our_turn or i % 4 == 0:
            fen = board.fen()
            try:
                _, osc, _, _ = eng.best_move(fen, ucis, 10**9, 10**9, 40, OUR_NODES)
                info = sf.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
                sfc = info["score"].pov(board.turn).score(mate_score=100000)
                if abs(osc) < 20000 and abs(sfc) < 20000:
                    rows.append((board.fullmove_number, osc, sfc, osc - sfc))
                    alldiffs.append(osc - sfc)
            except Exception:
                pass
        board.push(mv); ucis.append(mv.uci())
    if not rows:
        continue
    import statistics
    md = statistics.mean(abs(d) for _, _, _, d in rows)
    big = [(m, o, s) for m, o, s, d in rows if abs(d) > 200]
    print(f"{os.path.basename(path)[:38]:40s} n={len(rows):3d}  mean|our-SF|={md/100:.2f}  >2pawns off: {len(big)}")
    for m, o, s in big[:6]:
        tag = "OPTIMISTIC" if o > s else "pessimistic"
        print(f"     move {m:3d}   ours {o/100:+.2f}   SF {s/100:+.2f}   ({tag} by {abs(o-s)/100:.1f})")

import statistics
print(f"\nOVERALL  n={len(alldiffs)}  mean|diff|={statistics.mean(abs(d) for d in alldiffs)/100:.2f}  "
      f"mean signed={statistics.mean(alldiffs)/100:+.2f}  (+ = we're systematically optimistic)")
sf.quit()
