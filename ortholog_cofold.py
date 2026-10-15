#!/usr/bin/env python
"""
Ortholog co-fold: does a designed binder still form its complex with the orthologue?

Re-predicts each design against BOTH species' target sequence with Boltz-2 under
IDENTICAL settings, and reports the paired difference in iPTM. Needs no orthologue
structure (Boltz-2 co-folds from sequence), which matters here: there are zero PDB
entries for mouse EGFR.

Read the DELTA, never the absolute. iPTM is not comparable across different targets,
and the whole point is the paired comparison: same binder, same pocket constraint, same
MSA policy, one sequence changed.

HONEST LIMIT — read before quoting a number as evidence:
    Boltz-2/AF2-class models are weakly sensitive to point substitutions. Two sequences
    differing at a handful of positions routinely co-fold to near-identical complexes
    with near-identical iPTM even when binding is experimentally lost. So:
      * a LARGE drop is meaningful  -> the design likely fails on the orthologue
      * a SMALL delta is NOT evidence of cross-reactivity, only absence of evidence
        against it
    Pair it with the contact-footprint check (crossreactivity_filter.py), which says
    WHICH substituted residue the binder is touching when the two co-folds look alike.

Settings are copied from the design pipeline's own Boltz stage so the numbers stay
comparable with `boltz_iptm` in rankings.csv:
    * target chain A gets a real MSA (ColabFold server), cached per species
    * binder chain B gets `msa: empty` -- a designed sequence has no natural homologs,
      and a deep MSA overrides the pocket constraint
    * a pocket constraint steers docking to the epitope

Standalone: imports nothing from generate_binders.py.

Usage
-----
    python ortholog_cofold.py \\
        --rankings run/rankings.csv --top 40 \\
        --target_a sequences/human_EGFR_dIII.fasta --label_a human \\
        --target_b sequences/mouse_EGFR_dIII.fasta --label_b mouse \\
        --site "384,406,407,409,410,411,417,434,463,464,465" --target_start 311 \\
        --out_dir run/ortholog/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from config_loader import cfg
except ImportError:
    cfg = None

SEQ_COLS = ("binder_sequence", "sequence", "binder_seq", "seq")
ID_COLS = ("design_id", "name", "id", "design")


def log(m):
    print(m, file=sys.stderr, flush=True)


def read_fasta(p):
    return "".join(l.strip() for l in open(p) if not l.startswith(">"))


def load_designs(args):
    """[(design_id, binder_sequence)] from rankings.csv or a FASTA."""
    out = []
    if args.rankings:
        rows = list(csv.DictReader(open(args.rankings)))
        seq_col = next((c for c in SEQ_COLS if rows and c in rows[0]), None)
        id_col = next((c for c in ID_COLS if rows and c in rows[0]), None)
        if not seq_col:
            log(f"ERROR: no sequence column in {args.rankings}; looked for {SEQ_COLS}")
            return []
        for i, r in enumerate(rows):
            s = (r.get(seq_col) or "").strip().upper()
            if not s:
                continue
            if args.ranked_only and not (r.get("rank") or "").strip():
                continue
            out.append(((r.get(id_col) or f"design_{i:04d}").strip(), s))
    else:
        name, seq = None, []
        for line in open(args.fasta):
            line = line.strip()
            if line.startswith(">"):
                if name and seq:
                    out.append((name, "".join(seq).upper()))
                name, seq = line[1:].split()[0], []
            elif line:
                seq.append(line)
        if name and seq:
            out.append((name, "".join(seq).upper()))
    # de-duplicate identical sequences: co-folding the same sequence twice is wasted GPU
    seen, uniq = {}, []
    for n, s in out:
        if s in seen:
            continue
        seen[s] = n
        uniq.append((n, s))
    if len(uniq) < len(out):
        log(f"  {len(out) - len(uniq)} duplicate sequence(s) collapsed")
    return uniq[: args.top] if args.top else uniq


def boltz_cmd(*extra):
    conda = cfg.conda_run_args("boltz") if cfg is not None else ["-n", "boltz"]
    return ["conda", "run", "--no-capture-output", *conda, "boltz", "predict", *extra]


def target_msa(seq, label, work, devices):
    """Run Boltz once on the bare target to obtain and cache its MSA csv."""
    cache = work / f"msa_{label}.csv"
    if cache.exists():
        log(f"  [{label}] MSA cached")
        return cache
    d = work / f"msa_run_{label}"
    d.mkdir(parents=True, exist_ok=True)
    y = d / "target.yaml"
    y.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n"
                 f"      sequence: {seq}\n")
    cmd = boltz_cmd(str(y), "--out_dir", str(d), "--use_msa_server",
                    "--recycling_steps", "1", "--diffusion_samples", "1",
                    "--sampling_steps", "10")
    log(f"  [{label}] fetching target MSA (ColabFold server, rate-limited)...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    hits = sorted(d.glob("**/msa/*_0.csv"))
    if not hits:
        log(f"  [{label}] WARNING: no MSA produced; falling back to --use_msa_server per design")
        log((r.stderr or r.stdout or "")[-600:])
        return None
    cache.write_bytes(hits[0].read_bytes())
    log(f"  [{label}] MSA cached -> {cache}")
    return cache


def write_yaml(path, target_seq, binder_seq, msa_path, site_idx):
    y = ["version: 1", "sequences:",
         "  - protein:", "      id: A", f"      sequence: {target_seq}"]
    y.append(f"      msa: {msa_path}" if msa_path else "      msa: empty")
    y += ["  - protein:", "      id: B", f"      sequence: {binder_seq}",
          # designed sequences have no natural homologs, and a deep MSA would
          # override the pocket constraint -- the exact failure the design
          # pipeline's binder-validation mode exists to prevent
          "      msa: empty"]
    if site_idx:
        y += ["constraints:", "  - pocket:", "      binder: B", "      contacts:"]
        y += [f"        - [A, {i}]" for i in site_idx]
    path.write_text("\n".join(y) + "\n")


def parse_confidence(out_dir):
    """{stem: {iptm, ptm, plddt}} from Boltz confidence json."""
    res = {}
    for j in Path(out_dir).glob("**/confidence_*.json"):
        try:
            d = json.loads(j.read_text())
        except Exception:
            continue
        stem = j.stem.replace("confidence_", "")
        for suf in ("_model_0", "_model_1", "_model_2"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        prev = res.get(stem)
        cur = {"iptm": d.get("iptm"), "ptm": d.get("ptm"),
               "plddt": d.get("complex_plddt") or d.get("plddt")}
        if prev is None or (cur["iptm"] or -1) > (prev["iptm"] or -1):
            res[stem] = cur
    return res


def run_species(label, target_seq, designs, site_idx, work, devices, msa):
    d = work / f"cofold_{label}"
    ydir = d / "inputs"
    ydir.mkdir(parents=True, exist_ok=True)
    for name, seq in designs:
        write_yaml(ydir / f"{name}.yaml", target_seq, seq, msa, site_idx)
    cmd = boltz_cmd(str(ydir), "--out_dir", str(d),
                    "--recycling_steps", "3", "--diffusion_samples", "1",
                    "--sampling_steps", "200", "--preprocessing-threads", "24")
    if msa is None:
        cmd.append("--use_msa_server")
    env = dict(os.environ)
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices
    log(f"  [{label}] co-folding {len(designs)} complexes...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=86400, env=env)
    if r.returncode != 0:
        log(f"  [{label}] boltz exited {r.returncode}")
        log((r.stderr or r.stdout or "")[-2000:])
    return parse_confidence(d)


def main():
    ap = argparse.ArgumentParser(
        description="Paired Boltz-2 co-fold of designs against two species' targets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Read the DELTA, not the absolute. A small delta is not proof of "
               "cross-reactivity — see the limit noted at the top of this file.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rankings", help="rankings.csv from a design run")
    src.add_argument("--fasta", help="FASTA of binder sequences")
    ap.add_argument("--ranked_only", action="store_true",
                    help="skip rows with an empty rank (filtered-out designs)")
    ap.add_argument("--top", type=int, default=0, help="only the first N designs")
    ap.add_argument("--target_a", required=True, help="reference species target FASTA")
    ap.add_argument("--target_b", required=True, help="orthologue target FASTA")
    ap.add_argument("--label_a", default="human")
    ap.add_argument("--label_b", default="mouse")
    ap.add_argument("--site", default="",
                    help="epitope residues in the ORIGINAL target numbering, comma separated")
    ap.add_argument("--target_start", type=int, default=1,
                    help="residue number of the first residue in the target FASTA. The "
                         "Boltz pocket constraint takes 1-based TOKEN indices, not PDB "
                         "residue numbers, so a cropped target must declare its start "
                         "(domain III crop = 311). Getting this wrong silently steers "
                         "docking to the wrong patch.")
    ap.add_argument("--devices", default=None, help="value for CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out_dir", default="./ortholog_cofold")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ta, tb = read_fasta(args.target_a), read_fasta(args.target_b)
    if len(ta) != len(tb):
        log(f"ERROR: target lengths differ ({len(ta)} vs {len(tb)}). The two targets must be "
            f"aligned, equal-length sequences so one set of pocket indices applies to both.")
        return 1
    if "-" in ta or "-" in tb:
        log("ERROR: gap character in a target sequence — pocket indices would be shifted. "
            "Supply ungapped, equal-length orthologous regions.")
        return 1
    ndiff = sum(1 for x, y in zip(ta, tb) if x != y)

    site_idx = []
    if args.site:
        for tok in args.site.replace(" ", "").split(","):
            if not tok:
                continue
            n = int(tok)
            i = n - args.target_start + 1
            if not (1 <= i <= len(ta)):
                log(f"ERROR: site residue {n} maps to token {i}, outside the target "
                    f"(1..{len(ta)}). Check --target_start.")
                return 1
            site_idx.append(i)

    designs = load_designs(args)
    if not designs:
        log("ERROR: no designs loaded")
        return 1

    log(f"ortholog_cofold: {len(designs)} designs x 2 species")
    log(f"  targets {len(ta)} aa, {ndiff} substitutions between {args.label_a} and {args.label_b}")
    if site_idx:
        log(f"  pocket tokens {site_idx}  (from residues {args.site} with start {args.target_start})")
    if args.dry_run:
        log("  dry run — nothing executed")
        return 0

    work = Path(args.out_dir)
    work.mkdir(parents=True, exist_ok=True)
    msa_a = target_msa(ta, args.label_a, work, args.devices)
    msa_b = target_msa(tb, args.label_b, work, args.devices)

    ra = run_species(args.label_a, ta, designs, site_idx, work, args.devices, msa_a)
    rb = run_species(args.label_b, tb, designs, site_idx, work, args.devices, msa_b)

    rows = []
    for name, seq in designs:
        a, b = ra.get(name, {}), rb.get(name, {})
        ia, ib = a.get("iptm"), b.get("iptm")
        delta = (ib - ia) if (ia is not None and ib is not None) else None
        rows.append({
            "design_id": name, "length": len(seq),
            f"iptm_{args.label_a}": round(ia, 4) if ia is not None else "",
            f"iptm_{args.label_b}": round(ib, 4) if ib is not None else "",
            "delta_iptm": round(delta, 4) if delta is not None else "",
            "retained_frac": round(ib / ia, 3) if (ia and ib) else "",
            f"plddt_{args.label_a}": round(a.get("plddt") or 0, 2) if a else "",
            f"plddt_{args.label_b}": round(b.get("plddt") or 0, 2) if b else "",
        })
    rows.sort(key=lambda r: (r["delta_iptm"] if r["delta_iptm"] != "" else -9), reverse=True)

    out_csv = work / "ortholog_cofold.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'design':<30}{args.label_a+' iPTM':>13}{args.label_b+' iPTM':>13}{'delta':>9}{'kept':>8}")
    for r in rows[:30]:
        print(f"{r['design_id']:<30}{str(r[f'iptm_{args.label_a}']):>13}"
              f"{str(r[f'iptm_{args.label_b}']):>13}{str(r['delta_iptm']):>9}"
              f"{str(r['retained_frac']):>8}")
    print(f"\nwrote {out_csv}")
    print("\nREAD THE DELTA, NOT THE ABSOLUTE. A large drop means the design likely fails on")
    print(f"{args.label_b}. A small delta is NOT proof of cross-reactivity — Boltz-2 is weakly")
    print("sensitive to point substitutions. Cross-check with crossreactivity_filter.py to see")
    print("which substituted residue the binder actually touches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
