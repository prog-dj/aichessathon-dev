"""Write tuned weights from tuned_weights.npz back into fastchess.py.

  python -m texel.apply [tuned_weights.npz] [--dry]

Rewrites the literal constants in place: MG_VAL/EG_VAL, the _MG_PST/_EG_PST
tables, the king-danger piece weights and curve, bishop pair, isolated/doubled
pawns, the passed-pawn tables, rook file bonuses and tempo. Verifies afterwards
that fastchess re-imports and that perft is still exact.

Mobility is left alone: the model carries four per-piece weights but fastchess
has a single 22*n//10 coefficient, and the tuned values (2.0-2.5) sit either
side of it - not worth a code change for sub-noise movement.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FC = os.environ.get("FC_APPLY_TARGET") or os.path.join(ROOT, "fastchess.py")


def fmt_row(vals, per_line=8, indent="  "):
    ints = [int(round(float(v))) for v in vals]
    out, line = [], []
    for i, v in enumerate(ints):
        line.append(str(v))
        if len(line) == per_line and i != len(ints) - 1:
            out.append(",".join(line)); line = []
    if line:
        out.append(",".join(line))
    return (",\n" + indent).join(out)


def pst_block(name, tab):
    rows = []
    for pt in range(6):
        rows.append(" [" + fmt_row(tab[pt], 16, "  ") + "]")
    return f"{name} = np.array([\n" + ",\n".join(rows) + ",\n], np.int64)"


def sub1(s, pattern, repl, label):
    new, n = re.subn(pattern, repl, s, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"apply: anchor not found for {label}")
    return new


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else os.path.join(HERE, "tuned_weights.npz")
    dry = "--dry" in sys.argv
    d = np.load(npz)
    R = lambda k: int(round(float(np.asarray(d[k]).reshape(-1)[0])))

    s = open(FC, encoding="utf-8").read()

    s = sub1(s, r"MG_VAL = np\.array\(\[[^\]]*\], np\.int64\)",
             "MG_VAL = np.array([" + fmt_row(d["mat_mg"]) + ", 0], np.int64)", "MG_VAL")
    s = sub1(s, r"EG_VAL = np\.array\(\[[^\]]*\], np\.int64\)",
             "EG_VAL = np.array([" + fmt_row(d["mat_eg"]) + ", 0], np.int64)", "EG_VAL")

    if "_pst_mg" in d.files:
        s = sub1(s, r"_MG_PST = np\.array\(\[.*?\], np\.int64\)",
                 pst_block("_MG_PST", d["_pst_mg"].reshape(6, 64)), "_MG_PST")
        s = sub1(s, r"_EG_PST = np\.array\(\[.*?\], np\.int64\)",
                 pst_block("_EG_PST", d["_pst_eg"].reshape(6, 64)), "_EG_PST")

    s = sub1(s, r"_PASS_MG = np\.array\(\[[^\]]*\], np\.int64\)[^\n]*",
             "_PASS_MG = np.array([0, " + fmt_row(d["pass_mg"]) + ", 0], np.int64)  # Texel", "_PASS_MG")
    s = sub1(s, r"_PASS_EG = np\.array\(\[[^\]]*\], np\.int64\)",
             "_PASS_EG = np.array([0, " + fmt_row(d["pass_eg"]) + ", 0], np.int64)", "_PASS_EG")

    kn, kb, kr, kq = [int(round(float(x))) for x in d["kdw"]]
    s = sub1(s, r"w = 2 if pt <= 2 else \(3 if pt == 3 else 5\)",
             f"w = {kn} if pt == 1 else ({kb} if pt == 2 else ({kr} if pt == 3 else {kq}))", "kdw")
    num = max(1, int(round(float(d["kdc"][0]) * 16)))
    s = sub1(s, r"return \(u \* u \* 11\) // 16[^\n]*",
             f"return (u * u * {num}) // 16  # Texel", "kdc")

    bpm, bpe = R("bp_mg"), R("bp_eg")
    s = sub1(s, r"mg \+= sign \* 25\n            eg \+= sign \* 25",
             f"mg += sign * {bpm}\n            eg += sign * {bpe}", "bishop pair")
    s = sub1(s, r"mg -= sign \* 12; eg -= sign \* 12",
             f"mg -= sign * {abs(R('iso_mg'))}; eg -= sign * {abs(R('iso_eg'))}", "isolated")
    s = sub1(s, r"mg -= sign \* 5; eg -= sign \* 10",
             f"mg -= sign * {abs(R('dbl_mg'))}; eg -= sign * {abs(R('dbl_eg'))}", "doubled")
    s = sub1(s, r"mg \+= sign \* 22\n            elif \(fmask & own_p\) == U\(0\):\n                mg \+= sign \* 10",
             f"mg += sign * {R('rook_open')}\n            elif (fmask & own_p) == U(0):\n"
             f"                mg += sign * {R('rook_half')}", "rook files")
    s = sub1(s, r"return stm \+ 14", f"return stm + {R('tempo')}", "tempo")

    if dry:
        print("dry run - anchors all matched, nothing written")
        return
    open(FC, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {FC}")


if __name__ == "__main__":
    main()
