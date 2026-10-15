#!/usr/bin/env python
"""
Cross-species cross-reactivity filter for designed binders.

Question it answers: will this binder, designed against the human target, also bind the
orthologue (mouse, cyno, ...)? Without predicting anything against the orthologue.

The logic: if the binder's contact footprint lands only on residues that are IDENTICAL
between the two species, it binds the orthologue by construction. The failure mode is
footprint SPILL onto a substituted position. So this measures the footprint and reports
which substituted residues it touches -- CPU only, seconds per design, and it needs no
orthologue structure (which for mouse EGFR does not exist: zero PDB entries for Q01279).

THE NUMBERING FRAME IS DERIVED, NOT ASSUMED. The target PDB's observed sequence is
aligned to the reference FASTA, so the offset between PDB numbering and UniProt numbering
is measured and reported. This matters: the week-1 EGFR target is 6ARU, whose mature
numbering runs 24 lower than the UniProt frame the challenge quotes. Hardcoding an offset
is how an epitope silently moves by 24 residues.

Substitutions are split into `divergent` (BLOSUM62 <= 0, a real risk) and `conservative`
(BLOSUM62 > 0, usually tolerated) so a design is not thrown away for an I->M.

Standalone by design: imports nothing from generate_binders.py.

Usage
-----
    # what differs, in the target's own numbering -- works with no designs yet
    python crossreactivity_filter.py --target t.pdb --target_chain A \\
        --ref human.fasta --ortholog mouse.fasta --list_divergent

    # screen a finished run
    python crossreactivity_filter.py --target t.pdb --target_chain A \\
        --ref human.fasta --ortholog mouse.fasta \\
        --results_dir ./run/ --glob 'top_designs/*.pdb' --binder_chain B \\
        --out_csv crossreact.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V", "MSE": "M"}
CONTACT_CUT = 4.5


def read_fasta(path):
    seq = []
    for line in open(path):
        if not line.startswith(">"):
            seq.append(line.strip())
    return "".join(seq)


def as_pdb(path, _cache={}):
    """Return a .pdb path for `path`, converting mmCIF with gemmi.

    The pipeline writes top_designs/ as mmCIF, not PDB — a *.pdb glob over that
    directory silently matches only the target copy and every design is skipped while
    the run still reports success. Handle both suffixes here rather than relying on the
    caller's glob being right.
    """
    import tempfile
    path = Path(path)
    if path.suffix.lower() not in (".cif", ".mmcif"):
        return path
    key = str(path.resolve())
    if key in _cache:
        return _cache[key]
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.shorten_chain_names()
    d = Path(tempfile.mkdtemp(prefix="xreact_"))
    out = d / (path.stem + ".pdb")
    st.write_pdb(str(out))
    _cache[key] = out
    return out


def pdb_chain(path, chain):
    """{resnum: (resname1, {atom: xyz})} for one chain, ordered by residue number."""
    path = as_pdb(path)
    out = {}
    for line in open(path):
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        if line[16] not in (" ", "A"):
            continue
        rn = line[17:20].strip()
        if rn not in THREE2ONE:
            continue
        try:
            num = int(line[22:26])
        except ValueError:
            continue
        if line[26].strip():
            continue
        name = line[12:16].strip()
        if name.startswith("H") or line[76:78].strip() == "H":
            continue
        e = out.setdefault(num, [THREE2ONE[rn], {}])
        e[1][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return out


def map_pdb_to_ref(pdb_res, ref_seq):
    """Align the chain's observed sequence to the reference. Returns
    {pdb_resnum: ref_index0} plus the modal offset, so the frame is measured."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    nums = sorted(pdb_res)
    obs = "".join(pdb_res[n][0] for n in nums)
    al = PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                         open_gap_score=-11, extend_gap_score=-1)
    al.target_end_gap_score = al.query_end_gap_score = 0.0
    a = al.align(ref_seq, obs)[0]
    m = {}
    for (r0, r1), (q0, q1) in zip(*a.aligned):
        for k in range(r1 - r0):
            m[nums[q0 + k]] = r0 + k
    if not m:
        return {}, None, 0.0
    offs = [v + 1 - k for k, v in m.items()]          # uniprot(1-based) - pdbnum
    modal = max(set(offs), key=offs.count)
    ident = sum(1 for k, v in m.items() if pdb_res[k][0] == ref_seq[v]) / len(m)
    return m, modal, ident


def species_diff(pdb_res, ref_seq, orth_seq):
    """{pdb_resnum: dict} for every observed residue, classifying the orthologue."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    bl = substitution_matrices.load("BLOSUM62")
    pdb2ref, offset, ident = map_pdb_to_ref(pdb_res, ref_seq)
    al = PairwiseAligner(mode="global", substitution_matrix=bl,
                         open_gap_score=-11, extend_gap_score=-1)
    a = al.align(ref_seq, orth_seq)[0]
    ref2orth = {}
    ri = oi = 0
    for rc, oc in zip(a[0], a[1]):
        if rc != "-" and oc != "-":
            ref2orth[ri] = oi
        if rc != "-":
            ri += 1
        if oc != "-":
            oi += 1
    out = {}
    for num, ridx in pdb2ref.items():
        aa = pdb_res[num][0]
        oidx = ref2orth.get(ridx)
        oaa = orth_seq[oidx] if oidx is not None else None
        if oaa is None:
            cls = "gap"
        elif oaa == aa:
            cls = "identical"
        else:
            try:
                cls = "conservative" if bl[aa][oaa] > 0 else "divergent"
            except (KeyError, IndexError):
                cls = "divergent"
        out[num] = {"aa": aa, "ref_index1": ridx + 1, "ortholog": oaa, "class": cls}
    return out, offset, ident


def footprint(complex_pdb, binder_chain, target_chain, cutoff=CONTACT_CUT):
    """Target residue numbers contacted by the binder."""
    import numpy as np
    b = pdb_chain(complex_pdb, binder_chain)
    t = pdb_chain(complex_pdb, target_chain)
    if not b or not t:
        return None
    bc = np.asarray([c for v in b.values() for c in v[1].values()])
    hit = set()
    c2 = cutoff * cutoff
    for num, (_, atoms) in t.items():
        tc = np.asarray(list(atoms.values()))
        if not len(tc) or not len(bc):
            continue
        if ((tc[:, None, :] - bc[None, :, :]) ** 2).sum(-1).min() < c2:
            hit.add(num)
    return hit


def main():
    ap = argparse.ArgumentParser(
        description="Reject designs whose footprint touches species-substituted residues",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="target structure the designs were made against")
    ap.add_argument("--target_chain", default="A")
    ap.add_argument("--ref", required=True, help="reference-species FASTA (e.g. human UniProt)")
    ap.add_argument("--ortholog", required=True, help="orthologue FASTA (e.g. mouse)")
    ap.add_argument("--binder_chain", default="B")
    ap.add_argument("--complex", help="one complex to screen")
    ap.add_argument("--results_dir", help="a run directory to screen")
    ap.add_argument("--glob", default="top_designs/*.pdb")
    ap.add_argument("--contact_dist", type=float, default=CONTACT_CUT)
    ap.add_argument("--fail_on_conservative", action="store_true",
                    help="also fail a design for BLOSUM62-POSITIVE substitutions. Off by "
                         "default: those are usually tolerated, and failing on them would "
                         "reject most of the cetuximab-class epitope for an I->M.")
    ap.add_argument("--max_divergent", type=int, default=0,
                    help="how many divergent contacts a design may have and still pass "
                         "(default 0 = none)")
    ap.add_argument("--list_divergent", action="store_true",
                    help="just print the substituted positions in target numbering and exit")
    ap.add_argument("--restrict", default=None,
                    help="only consider this residue range in target numbering, e.g. 311-480")
    ap.add_argument("--out_csv")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tres = pdb_chain(args.target, args.target_chain)
    if not tres:
        print(f"ERROR: chain {args.target_chain!r} not found in {args.target}", file=sys.stderr)
        return 1
    diff, offset, ident = species_diff(tres, read_fasta(args.ref), read_fasta(args.ortholog))
    if args.restrict:
        lo, hi = (int(x) for x in args.restrict.split("-"))
        diff = {k: v for k, v in diff.items() if lo <= k <= hi}

    div = sorted(k for k, v in diff.items() if v["class"] == "divergent")
    conv = sorted(k for k, v in diff.items() if v["class"] == "conservative")
    same = sum(1 for v in diff.values() if v["class"] == "identical")

    print(f"# target {Path(args.target).name} chain {args.target_chain}: "
          f"{len(diff)} residues compared", file=sys.stderr)
    print(f"# NUMBERING FRAME MEASURED: reference number = PDB number + {offset}  "
          f"({ident:.1%} identity to the reference over the aligned region)", file=sys.stderr)
    print(f"# identical {same}  conservative {len(conv)}  DIVERGENT {len(div)}", file=sys.stderr)

    if args.list_divergent:
        print("\nDIVERGENT (BLOSUM62 <= 0) — reject footprints touching these")
        print("  target numbering: " + ",".join(map(str, div)))
        print("  reference frame : " + ",".join(str(diff[k]['ref_index1']) for k in div))
        for k in div:
            v = diff[k]
            print(f"    {v['aa']}{k} (ref {v['ref_index1']}) -> {v['ortholog']}")
        print("\nCONSERVATIVE (BLOSUM62 > 0) — usually tolerated")
        print("  target numbering: " + ",".join(map(str, conv)))
        for k in conv:
            v = diff[k]
            print(f"    {v['aa']}{k} (ref {v['ref_index1']}) -> {v['ortholog']}")
        return 0

    if args.complex:
        designs = [Path(args.complex)]
    elif args.results_dir:
        designs = sorted(Path(args.results_dir).glob(args.glob))
    else:
        ap.error("need --complex, --results_dir or --list_divergent")
    if not designs:
        print("ERROR: no design structures found", file=sys.stderr)
        return 1

    rows = []
    for d in designs:
        fp = footprint(d, args.binder_chain, args.target_chain, args.contact_dist)
        if fp is None:
            rows.append({"design": d.name, "error": "chain missing", "passes": False})
            continue
        hit_div = sorted(n for n in fp if diff.get(n, {}).get("class") == "divergent")
        hit_con = sorted(n for n in fp if diff.get(n, {}).get("class") == "conservative")
        bad = len(hit_div) + (len(hit_con) if args.fail_on_conservative else 0)
        rows.append({
            "design": d.name,
            "n_contacts": len(fp),
            "n_divergent_contacted": len(hit_div),
            "divergent_contacted": ",".join(f"{diff[n]['aa']}{n}>{diff[n]['ortholog']}" for n in hit_div),
            "n_conservative_contacted": len(hit_con),
            "conservative_contacted": ",".join(f"{diff[n]['aa']}{n}>{diff[n]['ortholog']}" for n in hit_con),
            "passes": bad <= args.max_divergent,
            "error": "",
        })

    npass = sum(1 for r in rows if r.get("passes"))
    if args.json:
        print(json.dumps({"offset": offset, "divergent_positions": div, "designs": rows},
                         indent=2, default=str))
    else:
        print(f"\n{'design':<44}{'contacts':>9}{'divergent':>10}  {'':4}")
        for r in sorted(rows, key=lambda x: (not x.get("passes"), x.get("n_divergent_contacted", 99))):
            print(f"{r['design']:<44}{r.get('n_contacts',0):>9}{r.get('n_divergent_contacted',0):>10}"
                  f"  {'PASS' if r.get('passes') else 'fail'}"
                  + (f"  {r.get('divergent_contacted','')}" if r.get("divergent_contacted") else ""))
        print(f"\n{npass}/{len(rows)} designs contact no divergent residue "
              f"(max_divergent={args.max_divergent})")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
