#!/usr/bin/env python
"""
pH-selectivity scoring for designed protein binders.

Answers: does this binder bind its target better at pH 6.5 than at pH 7.4?

Two independent signals are reported side by side, deliberately NOT blended:

  1. GEOMETRIC (primary).  Counts switch-competent histidines: an interface His whose
     side-chain nitrogen sits within H-bond distance of a carboxylate oxygen on the
     OTHER chain, plus the burial each interface His gains on binding.  This term does
     not depend on any pKa prediction, so it is the one to filter on.

  2. PROPKA LINKAGE (tie-breaker).  Thermodynamic linkage over the pKa values PROPKA
     assigns in the complex vs the separated partners:

         ln K(pH) = ln K_ref + Sum_complex ln(1+10^(pKa-pH)) - Sum_free ln(1+10^(pKa-pH))

     reported as `ph_selectivity_kcal` = dG(pH_high) - dG(pH_low); POSITIVE means
     tighter binding at the low pH, which is what this challenge rewards.

WHY THEY ARE NOT BLENDED: PROPKA's pKa RMSE is ~0.8 pH units and the whole 7.4 -> 6.5
window is 0.9 units.  The linkage number RANKS designs; it does not predict a
fold-change.  Quoting it as a predicted affinity ratio would be false precision.

The ceiling is physical, not a limitation of this script: each fully coupled titrating
group can buy at most 10^(pH_high - pH_low) ~= 8x over this window, and a single His at
the ideal pKa delivers ~2.8x.  Designs therefore need 2-3 coupled histidines.
`--self_test` checks the implementation reproduces both of those numbers.

Standalone by design: imports nothing from generate_binders.py, so it can be lifted into
its own package unchanged.  Needs propka + biopython (+ gemmi only for mmCIF inputs),
all present in the `boltz` host env.

Usage
-----
    python ph_scoring.py --complex design.pdb --binder_chain B --target_chain A
    python ph_scoring.py --results_dir ./run/ --binder_chain B --target_chain A --top 50
    python ph_scoring.py --self_test
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
R_KCAL = 0.0019872041          # kcal/(mol*K)
T_K = 298.15
RT = R_KCAL * T_K              # 0.59248 kcal/mol
LN10 = math.log(10.0)
RT_LN10 = RT * LN10            # 1.3642 kcal/mol per log10 unit

PH_LOW_DEFAULT = 6.5           # tumour microenvironment / challenge "binds"
PH_HIGH_DEFAULT = 7.4          # blood / challenge "must not bind"

# H-bond / salt-bridge geometry for the geometric term
HIS_N = ("ND1", "NE2")
ACID_O = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
HBOND_CUT = 4.0                # N...O, angstrom. Generous: PROPKA-free, so keep it inclusive.
IFACE_CUT = 4.5                # heavy-atom contact defining the interface

MAXASA = {"A": 106, "R": 248, "N": 157, "D": 163, "C": 135, "Q": 198, "E": 194,
          "G": 84, "H": 184, "I": 169, "L": 164, "K": 205, "M": 188, "F": 197,
          "P": 136, "S": 130, "T": 142, "W": 227, "Y": 222, "V": 142}
THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V", "MSE": "M"}
TITRATABLE = {"HIS", "ASP", "GLU", "LYS", "ARG", "CYS", "TYR"}


def _np():
    import numpy
    return numpy


# ── structure IO ─────────────────────────────────────────────────────────────

def to_pdb(path, out_dir=None):
    """Return a .pdb path for `path`, converting mmCIF with gemmi if needed.

    Boltz-2 writes mmCIF with multi-character chain IDs, which PROPKA (a fixed-column
    PDB reader) cannot represent.  gemmi's PDB writer renames them; we return the
    mapping so callers can translate their chain arguments.
    """
    path = Path(path)
    if path.suffix.lower() not in (".cif", ".mmcif"):
        return path, {}
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    before = [ch.name for ch in st[0]]
    st.shorten_chain_names()
    after = [ch.name for ch in st[0]]
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="ph_scoring_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + ".pdb")
    st.write_pdb(str(out))
    return out, dict(zip(before, after))


def write_chains(pdb_path, chains, out_path):
    """Write only `chains` (ATOM records, no HETATM) to out_path. Returns n atoms."""
    chains = set(chains)
    n = 0
    with open(out_path, "w") as o:
        for line in open(pdb_path):
            if line.startswith(("ATOM", "TER")) and line[21] in chains:
                o.write(line)
                n += 1 if line.startswith("ATOM") else 0
        o.write("END\n")
    return n


def load_residues(pdb_path, chains=None):
    """{(chain, resnum): {'name':3-letter, 'atoms':{atomname: (x,y,z)}}} for ATOM records."""
    out = {}
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        ch = line[21]
        if chains and ch not in chains:
            continue
        rn = line[17:20].strip()
        if rn not in THREE2ONE:
            continue
        try:
            num = int(line[22:26])
        except ValueError:
            continue
        if line[26].strip():                      # insertion code: skip, ambiguous numbering
            continue
        name = line[12:16].strip()
        if name.startswith("H") or line[76:78].strip() == "H":
            continue
        key = (ch, num)
        e = out.setdefault(key, {"name": rn, "atoms": {}})
        e["atoms"][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return out


# ── geometric term (primary signal) ──────────────────────────────────────────

def interface_residues(res, binder_chain, target_chain, cutoff=IFACE_CUT):
    """Residues of each chain with a heavy atom within `cutoff` of the other chain."""
    np = _np()
    b = {k: v for k, v in res.items() if k[0] == binder_chain}
    t = {k: v for k, v in res.items() if k[0] == target_chain}
    if not b or not t:
        return set(), set()
    tk = list(t)
    tcoord, towner = [], []
    for k in tk:
        for c in t[k]["atoms"].values():
            tcoord.append(c)
            towner.append(k)
    tcoord = np.asarray(tcoord)
    bi, ti = set(), set()
    c2 = cutoff * cutoff
    for k, v in b.items():
        bc = np.asarray(list(v["atoms"].values()))
        if not len(bc):
            continue
        d2 = ((bc[:, None, :] - tcoord[None, :, :]) ** 2).sum(-1)
        hit = np.where(d2.min(0) < c2)[0]
        if len(hit):
            bi.add(k)
            ti.update(towner[i] for i in hit)
    return bi, ti


def switch_pairs(res, binder_chain, target_chain, cutoff=HBOND_CUT):
    """Cross-chain His(N)...carboxylate(O) pairs — the pH-switch handles.

    Direction is recorded because it matters for design: 'binder_his' means the binder
    carries the titrating group; 'target_his' means we are exploiting a His on the
    target (e.g. EGFR H409), which costs the design nothing.
    """
    np = _np()
    pairs = []
    for hch, ach, label in ((binder_chain, target_chain, "binder_his"),
                            (target_chain, binder_chain, "target_his")):
        his = {k: v for k, v in res.items() if k[0] == hch and v["name"] == "HIS"}
        acid = {k: v for k, v in res.items() if k[0] == ach and v["name"] in ACID_O}
        for hk, hv in his.items():
            for an in HIS_N:
                if an not in hv["atoms"]:
                    continue
                hp = np.asarray(hv["atoms"][an])
                for ak, av in acid.items():
                    for ao in ACID_O[av["name"]]:
                        if ao not in av["atoms"]:
                            continue
                        d = float(np.linalg.norm(hp - np.asarray(av["atoms"][ao])))
                        if d <= cutoff:
                            pairs.append({
                                "kind": label,
                                "his": f"{hch}{hk[1]}", "his_atom": an,
                                "acid": f"{av['name']}{ach}{ak[1]}", "acid_atom": ao,
                                "dist": round(d, 2),
                            })
    # keep only the closest contact per (his, acid) pair
    best = {}
    for p in pairs:
        key = (p["his"], p["acid"])
        if key not in best or p["dist"] < best[key]["dist"]:
            best[key] = p
    return sorted(best.values(), key=lambda p: p["dist"])


def his_burial(complex_pdb, binder_pdb, target_pdb, binder_chain, target_chain):
    """Relative-SASA lost by each His on binding. Burial is what shifts a His pKa."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    p, sr = PDBParser(QUIET=True), ShrakeRupley()

    def rsa(path):
        st = p.get_structure("x", path)
        sr.compute(st[0], level="R")
        out = {}
        for ch in st[0]:
            for r in ch:
                rn = r.get_resname()
                if rn in THREE2ONE:
                    out[(ch.id, r.id[1])] = r.sasa / MAXASA[THREE2ONE[rn]]
        return out

    with open(os.devnull, "w") as dn, redirect_stderr(dn):
        cx = rsa(complex_pdb)
        free = {}
        free.update(rsa(binder_pdb))
        free.update(rsa(target_pdb))
    out = {}
    for k, v in free.items():
        if k in cx:
            out[k] = {"rsa_free": round(v, 3), "rsa_bound": round(cx[k], 3),
                      "d_rsa": round(v - cx[k], 3)}
    return out


# ── PROPKA linkage term (tie-breaker) ────────────────────────────────────────

def propka_pkas(pdb_path):
    """{(chain, resnum, resname): pKa} from PROPKA. Returns {} if PROPKA fails.

    PROPKA chatters missing-atom warnings on stdout for any incomplete residue, which
    is normal for predicted models; it is captured so it cannot pollute a pipeline log.
    """
    import propka.run as pr
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            mol = pr.single(str(pdb_path), optargs=["--quiet"], write_pka=False)
    except Exception as e:                                   # noqa: BLE001 - never fatal
        return {}, f"{type(e).__name__}: {e}"
    conf = mol.conformations.get("AVR") or next(iter(mol.conformations.values()))
    out = {}
    for g in conf.groups:
        pka = getattr(g, "pka_value", None)
        a = getattr(g, "atom", None)
        if pka is None or a is None:
            continue
        if a.res_name not in TITRATABLE:
            continue
        out[(a.chain_id, int(a.res_num), a.res_name)] = float(pka)
    return out, None


# Physically plausible pKa ranges. PROPKA will happily return a Glu at 9.1 for a buried
# carboxylate with no compensating charge; on the 6ARU control one such artifact
# (GLU C105, 4.52 -> 9.12) supplied the entire linkage score on its own. Shifts beyond
# MAX_DPKA are dropped from the sum and counted, rather than being allowed to dominate.
MAX_DPKA = 3.0
PLAUSIBLE_PKA = {"ASP": (1.0, 8.0), "GLU": (1.0, 8.0), "HIS": (2.0, 10.0),
                 "LYS": (6.0, 14.0), "ARG": (9.0, 16.0), "CYS": (3.0, 13.0),
                 "TYR": (7.0, 14.0)}


def linkage(pka_complex, pka_free, ph_low, ph_high, restrict=None, max_dpka=MAX_DPKA):
    """dG(ph_high) - dG(ph_low) in kcal/mol. POSITIVE = tighter at ph_low.

    Only groups present in BOTH states contribute; a group seen in one state only has
    no defined shift and is dropped rather than silently treated as unshifted.

    Returns (selectivity, contributors, n_coupled, n_rejected).
    """
    keys = set(pka_complex) & set(pka_free)
    if restrict is not None:
        keys &= set(restrict)

    rejected = []
    if max_dpka is not None:
        keep = set()
        for k in keys:
            lo, hi = PLAUSIBLE_PKA.get(k[2], (-99, 99))
            if abs(pka_complex[k] - pka_free[k]) > max_dpka:
                rejected.append((k, pka_free[k], pka_complex[k], "shift"))
            elif not (lo <= pka_complex[k] <= hi and lo <= pka_free[k] <= hi):
                rejected.append((k, pka_free[k], pka_complex[k], "range"))
            else:
                keep.add(k)
        keys = keep

    def S(ph):
        s = 0.0
        for k in keys:
            s += math.log1p(10.0 ** (pka_complex[k] - ph))
            s -= math.log1p(10.0 ** (pka_free[k] - ph))
        return s

    sel = RT * (S(ph_low) - S(ph_high))
    contrib = []
    for k in keys:
        d = RT * ((math.log1p(10.0 ** (pka_complex[k] - ph_low))
                   - math.log1p(10.0 ** (pka_free[k] - ph_low)))
                  - (math.log1p(10.0 ** (pka_complex[k] - ph_high))
                     - math.log1p(10.0 ** (pka_free[k] - ph_high))))
        if abs(d) >= 0.01:
            contrib.append({"residue": f"{k[2]}{k[0]}{k[1]}",
                            "pka_free": round(pka_free[k], 2),
                            "pka_bound": round(pka_complex[k], 2),
                            "d_pka": round(pka_complex[k] - pka_free[k], 2),
                            "kcal": round(d, 3)})
    contrib.sort(key=lambda c: -abs(c["kcal"]))
    return sel, contrib, len(keys), rejected


# ── top-level scoring ────────────────────────────────────────────────────────

def score_complex(complex_path, binder_chain="B", target_chain="A",
                  ph_low=PH_LOW_DEFAULT, ph_high=PH_HIGH_DEFAULT,
                  workdir=None, target_pka_cache=None, skip_propka=False):
    """Score one complex. Never raises: failures come back in the 'error' field."""
    res_out = {
        "complex": str(complex_path), "binder_chain": binder_chain,
        "target_chain": target_chain, "ph_low": ph_low, "ph_high": ph_high,
        "n_his_interface": 0, "n_switch_his": 0, "n_switch_binder_his": 0,
        "n_switch_target_his": 0, "his_burial_max": float("nan"),
        "switch_pairs": [], "ph_selectivity_kcal": float("nan"),
        "ph_fold_change": float("nan"), "n_coupled_groups": 0,
        "n_rejected_pka": 0, "rejected_pka": [], "dominant_fraction": float("nan"),
        "top_contributors": [], "error": None,
    }
    tmp = Path(workdir or tempfile.mkdtemp(prefix="ph_scoring_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        pdb, remap = to_pdb(complex_path, tmp)
        binder_chain = remap.get(binder_chain, binder_chain)
        target_chain = remap.get(target_chain, target_chain)
        res_out["binder_chain"], res_out["target_chain"] = binder_chain, target_chain

        res = load_residues(pdb)
        if not any(k[0] == binder_chain for k in res):
            res_out["error"] = f"binder chain {binder_chain!r} absent (have {sorted({k[0] for k in res})})"
            return res_out
        if not any(k[0] == target_chain for k in res):
            res_out["error"] = f"target chain {target_chain!r} absent (have {sorted({k[0] for k in res})})"
            return res_out

        # ---- geometric term
        bi, ti = interface_residues(res, binder_chain, target_chain)
        iface = bi | ti
        res_out["n_interface_res"] = len(iface)
        res_out["n_his_interface"] = sum(1 for k in iface if res[k]["name"] == "HIS")
        pairs = [p for p in switch_pairs(res, binder_chain, target_chain)]
        res_out["switch_pairs"] = pairs
        res_out["n_switch_his"] = len({p["his"] for p in pairs})
        res_out["n_switch_binder_his"] = len({p["his"] for p in pairs if p["kind"] == "binder_his"})
        res_out["n_switch_target_his"] = len({p["his"] for p in pairs if p["kind"] == "target_his"})

        bpdb, tpdb = tmp / "binder.pdb", tmp / "target.pdb"
        write_chains(pdb, [binder_chain], bpdb)
        write_chains(pdb, [target_chain], tpdb)

        try:
            bur = his_burial(pdb, bpdb, tpdb, binder_chain, target_chain)
            hb = [v["d_rsa"] for k, v in bur.items()
                  if k in res and res[k]["name"] == "HIS" and k in iface]
            res_out["his_burial_max"] = round(max(hb), 3) if hb else float("nan")
        except Exception as e:                               # noqa: BLE001
            res_out["error"] = f"sasa: {type(e).__name__}: {e}"

        if skip_propka:
            return res_out

        # ---- PROPKA linkage term
        pc, err_c = propka_pkas(pdb)
        if err_c:
            res_out["error"] = f"propka(complex): {err_c}"
            return res_out
        pb, err_b = propka_pkas(bpdb)
        if target_pka_cache is not None and "pka" in target_pka_cache:
            pt, err_t = target_pka_cache["pka"], None
        else:
            pt, err_t = propka_pkas(tpdb)
            if target_pka_cache is not None and not err_t:
                target_pka_cache["pka"] = pt
        if err_b or err_t:
            res_out["error"] = f"propka(free): {err_b or err_t}"
            return res_out
        free = dict(pt)
        free.update(pb)

        sel, contrib, ncoup, rej = linkage(pc, free, ph_low, ph_high)
        res_out["ph_selectivity_kcal"] = round(sel, 3)
        res_out["ph_fold_change"] = round(10.0 ** (sel / RT_LN10), 2)
        res_out["n_coupled_groups"] = ncoup
        res_out["n_rejected_pka"] = len(rej)
        res_out["rejected_pka"] = [
            {"residue": f"{k[2]}{k[0]}{k[1]}", "pka_free": round(f, 2),
             "pka_bound": round(b, 2), "why": w} for k, f, b, w in rej[:6]]
        res_out["top_contributors"] = contrib[:8]
        # A score carried by one residue is a red flag, not a result.
        if contrib and abs(sel) > 1e-6:
            res_out["dominant_fraction"] = round(abs(contrib[0]["kcal"]) / abs(sel), 2)
    except Exception as e:                                   # noqa: BLE001
        res_out["error"] = f"{type(e).__name__}: {e}"
    return res_out


# ── self test ────────────────────────────────────────────────────────────────

def self_test():
    """Check the linkage maths against the two values it must reproduce."""
    ok = True

    # 1. One His, pKa 6.5 free -> 7.5 bound, should give ~2.7x (the textbook single-site value)
    sel, _, _, _ = linkage({("B", 1, "HIS"): 7.5}, {("B", 1, "HIS"): 6.5}, 6.5, 7.4)
    fold = 10.0 ** (sel / RT_LN10)
    print(f"  single His (dpKa +1.0): {sel:+.3f} kcal/mol, {fold:.2f}x")
    if not (2.0 < fold < 3.5):
        print("    FAIL: expected ~2.5-3x"); ok = False

    # 2. Physical ceiling: one fully coupled proton cannot exceed 10^(7.4-6.5) = 7.94x
    sel, _, _, _ = linkage({("B", 1, "HIS"): 30.0}, {("B", 1, "HIS"): -20.0}, 6.5, 7.4, max_dpka=None)
    fold = 10.0 ** (sel / RT_LN10)
    print(f"  one proton, fully coupled (ceiling): {fold:.2f}x  (theory 10^0.9 = {10**0.9:.2f}x)")
    if abs(fold - 10 ** 0.9) > 0.05:
        print("    FAIL: ceiling not reproduced"); ok = False

    # 3. Three such histidines -> ceiling cubed
    pc = {("B", i, "HIS"): 30.0 for i in (1, 2, 3)}
    pf = {("B", i, "HIS"): -20.0 for i in (1, 2, 3)}
    sel, _, _, _ = linkage(pc, pf, 6.5, 7.4, max_dpka=None)
    fold = 10.0 ** (sel / RT_LN10)
    print(f"  three coupled protons: {fold:.1f}x  (theory {10**2.7:.1f}x)")
    if abs(fold - 10 ** 2.7) > 5:
        print("    FAIL"); ok = False

    # 4. No shift -> exactly zero selectivity
    sel, _, _, _ = linkage({("B", 1, "HIS"): 6.9}, {("B", 1, "HIS"): 6.9}, 6.5, 7.4)
    print(f"  unshifted His (must be 0): {sel:+.4f} kcal/mol")
    if abs(sel) > 1e-9:
        print("    FAIL: unshifted group must contribute nothing"); ok = False

    # 5. Wrong-direction shift must score negative
    sel, _, _, _ = linkage({("B", 1, "HIS"): 5.5}, {("B", 1, "HIS"): 6.5}, 6.5, 7.4)
    print(f"  reversed shift (must be < 0): {sel:+.3f} kcal/mol")
    if sel >= 0:
        print("    FAIL: sign convention broken"); ok = False

    print("\n  " + ("ALL OK — positive score = tighter at the LOW pH" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="pH-selectivity scoring for designed binders (geometric + PROPKA linkage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Positive ph_selectivity_kcal = binds more tightly at the LOW pH.")
    ap.add_argument("--complex", help="one complex structure (.pdb or .cif)")
    ap.add_argument("--results_dir", help="score every complex under a pipeline run dir")
    ap.add_argument("--glob", default="top_designs/*.pdb",
                    help="pattern under --results_dir (default: top_designs/*.pdb)")
    ap.add_argument("--binder_chain", default="B")
    ap.add_argument("--target_chain", default="A")
    ap.add_argument("--ph_low", type=float, default=PH_LOW_DEFAULT)
    ap.add_argument("--ph_high", type=float, default=PH_HIGH_DEFAULT)
    ap.add_argument("--top", type=int, default=0, help="score only the first N (0 = all)")
    ap.add_argument("--skip_propka", action="store_true",
                    help="geometric term only — instant, no pKa prediction")
    ap.add_argument("--out_csv", help="write a CSV of per-design scores")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        print("ph_scoring.py self test — linkage maths")
        return self_test()

    targets = []
    if args.complex:
        targets = [Path(args.complex)]
    elif args.results_dir:
        targets = sorted(Path(args.results_dir).glob(args.glob))
        if not targets:
            print(f"ERROR: no structures matched {args.results_dir}/{args.glob}", file=sys.stderr)
            return 1
    else:
        ap.error("need --complex, --results_dir or --self_test")
    if args.top:
        targets = targets[:args.top]

    cache, rows = {}, []
    for i, t in enumerate(targets, 1):
        r = score_complex(t, args.binder_chain, args.target_chain,
                          args.ph_low, args.ph_high,
                          target_pka_cache=cache, skip_propka=args.skip_propka)
        rows.append(r)
        if not args.json and len(targets) > 1:
            print(f"  [{i}/{len(targets)}] {Path(t).name:<44} "
                  f"switch_his={r['n_switch_his']} "
                  f"sel={r['ph_selectivity_kcal']} kcal "
                  f"({r['ph_fold_change']}x)"
                  + (f"  ERROR {r['error']}" if r["error"] else ""), file=sys.stderr)

    if args.json:
        print(json.dumps(rows if len(rows) > 1 else rows[0], indent=2, default=str))
    else:
        if len(rows) == 1:
            r = rows[0]
            print(f"\n{'='*66}\n{Path(r['complex']).name}\n{'='*66}")
            print(f"  interface residues     : {r.get('n_interface_res', 0)}")
            print(f"  His at interface       : {r['n_his_interface']}")
            print(f"  SWITCH-COMPETENT His   : {r['n_switch_his']} "
                  f"(binder {r['n_switch_binder_his']}, target {r['n_switch_target_his']})")
            for p in r["switch_pairs"]:
                print(f"      {p['his']}.{p['his_atom']} ... {p['acid']}.{p['acid_atom']}"
                      f"  {p['dist']} A   [{p['kind']}]")
            print(f"  max His burial (drsa)  : {r['his_burial_max']}")
            print(f"  PROPKA linkage         : {r['ph_selectivity_kcal']} kcal/mol "
                  f"-> {r['ph_fold_change']}x tighter at pH {r['ph_low']}")
            print(f"  coupled groups         : {r['n_coupled_groups']}"
                  f"   (rejected as unphysical: {r['n_rejected_pka']})")
            if r.get("dominant_fraction") == r.get("dominant_fraction") and r.get("dominant_fraction", 0) > 0.5:
                print(f"  WARNING: {r['dominant_fraction']:.0%} of the linkage score comes from a"
                      f" single residue — treat it as noise, not signal")
            for x in r.get("rejected_pka", []):
                print(f"      rejected {x['residue']:<12} {x['pka_free']} -> {x['pka_bound']}  ({x['why']})")
            for c in r["top_contributors"]:
                print(f"      {c['residue']:<12} pKa {c['pka_free']:>5} -> {c['pka_bound']:<5} "
                      f"(d {c['d_pka']:+.2f})  {c['kcal']:+.3f} kcal")
            if r["error"]:
                print(f"  ERROR: {r['error']}")
            print("\n  NOTE: PROPKA RMSE (~0.8 pH units) is comparable to the whole 6.5-7.4")
            print("  window. Treat the linkage number as a RANKING signal, not a predicted")
            print("  fold-change. Filter on switch-competent His count.")
        else:
            ok = [r for r in rows if not r["error"]]
            print(f"\nscored {len(rows)} complexes, {len(rows)-len(ok)} errors")
            ok.sort(key=lambda r: (-r["n_switch_his"],
                                   -(r["ph_selectivity_kcal"] if r["ph_selectivity_kcal"] == r["ph_selectivity_kcal"] else -9)))
            print(f"\n{'design':<44}{'switchHis':>10}{'kcal':>9}{'fold':>8}")
            for r in ok[:25]:
                print(f"{Path(r['complex']).name:<44}{r['n_switch_his']:>10}"
                      f"{r['ph_selectivity_kcal']:>9}{r['ph_fold_change']:>8}")

    if args.out_csv:
        cols = ["complex", "binder_chain", "target_chain", "n_interface_res",
                "n_his_interface", "n_switch_his", "n_switch_binder_his",
                "n_switch_target_his", "his_burial_max", "ph_selectivity_kcal",
                "ph_fold_change", "n_coupled_groups", "n_rejected_pka",
                "dominant_fraction", "error"]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
