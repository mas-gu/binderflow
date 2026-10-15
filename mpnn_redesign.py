#!/usr/bin/env python
"""
Interface-targeted histidine redesign of designed binders, via LigandMPNN.

Purpose: turn an affinity-optimised binder into a pH-switchable one by re-designing
ONLY its interface positions with a positive logit bias toward histidine, keeping the
rest of the sequence fixed so the validated fold is preserved.

Why per-position and not `--bias_AA H:2.0` globally: a global bias puts histidine in the
hydrophobic core, which is destabilising and aggregation-prone. Histidine is wanted at
the interface and nowhere else.

WHAT THIS DOES NOT DO — read this before trusting the output:

  * LigandMPNN is pH-blind and pKa-blind. It places histidine where the BACKBONE
    tolerates one, not where protonation is coupled to binding. A histidine whose pKa
    does not shift on binding contributes exactly zero pH selectivity.
    -> every variant must be scored afterwards with `ph_scoring.py`.

  * A redesigned sequence is NOT guaranteed to fold to the backbone it was designed on.
    -> every variant must be re-folded (Boltz-2) and RMSD-checked before it counts.

So this script is step 1 of 3: redesign -> re-fold/validate -> ph_scoring.

Standalone by design: imports nothing from generate_binders.py. Reads tool paths from
config.yaml through config_loader, the same way the pipeline does.

Usage
-----
    # one complex, look at what would be redesigned
    python mpnn_redesign.py --complex design.pdb --binder_chain B --target_chain A --dry_run

    # a whole directory of complexes, biased toward His at the interface
    python mpnn_redesign.py --complex_dir ./run/top_designs --binder_chain B \\
        --target_chain A --site "A:384,406,409,411,434" --his_bias 2.5 \\
        --n_seqs 8 --out_dir ./his_variants/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from config_loader import cfg
except ImportError:                                    # usable outside the repo too
    cfg = None

THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V", "MSE": "M"}
IFACE_CUT = 5.0
# Positions we refuse to mutate to His even when they are at the interface: Gly often
# carries a backbone conformation no side chain can adopt, Pro breaks the chain, and Cys
# is reserved for disulfides.
NO_TOUCH = {"GLY", "PRO", "CYS"}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_residues(pdb_path, chains=None):
    out = {}
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A"):
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
        if line[26].strip():
            continue
        name = line[12:16].strip()
        if name.startswith("H") or line[76:78].strip() == "H":
            continue
        e = out.setdefault((ch, num), {"name": rn, "atoms": {}})
        e["atoms"][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return out


def parse_site(site):
    """'A:384,406-409' -> (chain, {resnums}). Same grammar the pipeline uses."""
    if not site:
        return None, set()
    chain, _, spec = site.partition(":")
    chain = chain.strip()
    nums = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            nums.update(range(int(a), int(b) + 1))
        else:
            nums.add(int(part))
    return chain, nums


def binder_interface(res, binder_chain, target_chain, cutoff=IFACE_CUT,
                     site_chain=None, site_nums=None):
    """Binder positions contacting the target — optionally only those facing the site.

    Returns [(resnum, resname), ...] sorted. Restricting to the site matters: biasing
    the whole interface scatters histidines over contacts we do not care about, while
    the epitope-facing subset is where a pKa shift can actually couple to binding.
    """
    import numpy as np
    b = {k: v for k, v in res.items() if k[0] == binder_chain}
    tsel = {k: v for k, v in res.items() if k[0] == target_chain}
    if site_nums:
        sc = site_chain or target_chain
        narrowed = {k: v for k, v in tsel.items() if k[0] == sc and k[1] in site_nums}
        if narrowed:
            tsel = narrowed
        else:
            # Hard failure, not a fallback. Silently widening to the whole target changes
            # what gets designed while every log line still reads like success — and the
            # cause is almost always a NUMBERING FRAME mismatch, not a typo. Predicted
            # complexes are renumbered from 1 by their producer, so a site expressed in
            # the original target's numbering matches nothing in them.
            have = sorted(k[1] for k in tsel if k[0] == sc)
            rng = f"{have[0]}-{have[-1]}" if have else "none"
            raise SystemExit(
                f"ERROR: none of the --site residues {sorted(site_nums)[:6]}... exist in "
                f"chain {sc}, which is numbered {rng}.\n"
                f"  This is a numbering-frame mismatch. Predicted complexes (Boltz-2, and "
                f"anything cropped) are renumbered from 1, so a site written in the "
                f"original target's frame will not match.\n"
                f"  Fix with --site_offset: the offset to ADD to your site numbers to reach "
                f"this file's frame. For a crop starting at residue N, that is -(N-1).\n"
                f"  Refusing to silently redesign the whole interface instead.")
    if not b or not tsel:
        return []
    tc = np.asarray([c for v in tsel.values() for c in v["atoms"].values()])
    out = []
    c2 = cutoff * cutoff
    for k, v in sorted(b.items()):
        bc = np.asarray(list(v["atoms"].values()))
        if not len(bc):
            continue
        if ((bc[:, None, :] - tc[None, :, :]) ** 2).sum(-1).min() < c2:
            out.append((k[1], v["name"]))
    return out


def ligandmpnn_paths():
    d = ckpt = ""
    if cfg is not None:
        d = cfg.tool("ligandmpnn_dir", "LIGANDMPNN_DIR", "")
        ckpt = cfg.tool("ligandmpnn_ckpt", "LIGANDMPNN_CKPT", "")
    d = os.environ.get("LIGANDMPNN_DIR", d)
    ckpt = os.environ.get("LIGANDMPNN_CKPT", ckpt)
    return d, ckpt


def build_inputs(complexes, binder_chain, target_chain, his_bias, cutoff,
                 site_chain, site_nums, redesign_scope, work):
    """Write the bias / redesign JSONs LigandMPNN reads. Returns (json paths, report)."""
    bias_multi, redes_multi, report = {}, {}, []
    for pdb in complexes:
        res = load_residues(pdb)
        chains = sorted({k[0] for k in res})
        if binder_chain not in chains or target_chain not in chains:
            report.append({"pdb": str(pdb), "skipped": f"chains {chains}"})
            continue
        iface = binder_interface(res, binder_chain, target_chain, cutoff,
                                 site_chain, site_nums)
        mutable = [(n, rn) for n, rn in iface if rn not in NO_TOUCH]
        if not mutable:
            report.append({"pdb": str(pdb), "skipped": "no mutable interface position"})
            continue
        key = str(Path(pdb).resolve())
        bias_multi[key] = {f"{binder_chain}{n}": {"H": float(his_bias)} for n, _ in mutable}
        if redesign_scope == "interface":
            redes_multi[key] = " ".join(f"{binder_chain}{n}" for n, _ in mutable)
        # chain order as LigandMPNN will emit it (':'-separated, PDB order), plus the
        # index of each biased residue WITHIN the binder segment -- collect() needs both
        # to read the right part of the output line.
        seen, chain_order = set(), []
        for k in res:
            if k[0] not in seen:
                seen.add(k[0]); chain_order.append(k[0])
        bnums = sorted(k[1] for k in res if k[0] == binder_chain)
        binder_index = {str(n): i for i, n in enumerate(bnums)}
        report.append({
            "pdb": str(pdb), "n_interface": len(iface), "n_biased": len(mutable),
            "positions": [f"{rn}{n}" for n, rn in mutable],
            "already_his": [f"{rn}{n}" for n, rn in mutable if rn == "HIS"],
            "chain_order": chain_order,
            "binder_index": {str(n): binder_index[str(n)] for n, _ in mutable
                             if str(n) in binder_index},
        })
    if not bias_multi:
        return None, None, None, report
    pdb_list = work / "pdb_list.json"
    pdb_list.write_text(json.dumps({k: "" for k in bias_multi}))
    bias_path = work / "bias_per_residue.json"
    bias_path.write_text(json.dumps(bias_multi, indent=1))
    redes_path = None
    if redes_multi:
        redes_path = work / "redesigned_residues.json"
        redes_path.write_text(json.dumps(redes_multi, indent=1))
    return pdb_list, bias_path, redes_path, report


def run_mpnn(pdb_list, bias_path, redes_path, binder_chain, out_dir, n_seqs,
             temperature, seed, dry_run=False, timeout=7200, pack=True, cpu=False):
    d, ckpt = ligandmpnn_paths()
    if not d or not Path(d).is_dir():
        log(f"ERROR: LigandMPNN dir not found ({d!r}). Set LIGANDMPNN_DIR or config.yaml.")
        return 1, None
    conda_args = cfg.conda_run_args("mpnn") if cfg is not None else ["-n", "mpnn"]
    cmd = ["conda", "run", "--no-capture-output", *conda_args,
           "python", f"{d}/run.py",
           "--model_type", "protein_mpnn",
           "--pdb_path_multi", str(pdb_list),
           "--out_folder", str(out_dir),
           "--chains_to_design", binder_chain,
           "--bias_AA_per_residue_multi", str(bias_path),
           "--number_of_batches", str(n_seqs),
           "--batch_size", "1",
           "--temperature", str(temperature),
           "--seed", str(seed),
           "--save_stats", "1"]
    if pack:
        # Full-atom models let ph_scoring.py run straight away: its geometric term needs
        # side-chain N and O atoms, which a backbone-only output does not have. This is
        # a cheap triage step, NOT a substitute for re-folding -- packing places side
        # chains on the ORIGINAL backbone and cannot tell you the sequence still folds.
        cmd += ["--pack_side_chains", "1", "--number_of_packs_per_design", "1",
                "--repack_everything", "0"]
        # --checkpoint_path_sc defaults to a path relative to the LigandMPNN install
        # that is not populated here; point it at the weights dir beside the MPNN
        # checkpoint, which is where this box keeps it.
        sc = os.environ.get("LIGANDMPNN_SC_CKPT", "")
        if not sc and ckpt:
            cand = Path(ckpt).parent / "ligandmpnn_sc_v_32_002_16.pt"
            if cand.is_file():
                sc = str(cand)
        if sc:
            cmd += ["--checkpoint_path_sc", sc]
        else:
            log("  NOTE: side-chain packing checkpoint not found; packing may fail")
    if ckpt:
        cmd[cmd.index("--model_type"):cmd.index("--model_type")] = []
        cmd += ["--checkpoint_protein_mpnn", ckpt]
    if redes_path:
        cmd += ["--redesigned_residues_multi", str(redes_path)]
    log("  " + " ".join(cmd))
    if dry_run:
        return 0, cmd
    # LigandMPNN has no --device flag: run.py does
    #   torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # so the only way to keep it off the card is to hide the card. Worth doing when a
    # long design campaign owns the GPU — ProteinMPNN is small enough that CPU costs
    # little, and this way the His redesign runs ALONGSIDE the campaign, not behind it.
    env = dict(os.environ)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        log("  running on CPU (CUDA_VISIBLE_DEVICES='') — leaves the GPU free")
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        log(f"ERROR: LigandMPNN exited {r.returncode}")
        log((r.stderr or r.stdout or "")[-2500:])
        return r.returncode, cmd
    return 0, cmd


def collect(out_dir, report, binder_chain):
    """Read the FASTAs back and count His AT THE BIASED POSITIONS.

    LigandMPNN writes every chain on one line, ':'-separated, in the order the chains
    appear in the input PDB -- so the binder is a SEGMENT of that line, not the line.
    Reading the whole line as the binder silently reports the target's histidines and
    makes every variant look identical. The first record in each file is the input
    sequence, kept and flagged as `is_input` so the diff is visible.
    """
    seqs_dir = Path(out_dir) / "seqs"
    by_stem = {Path(e["pdb"]).stem: e for e in report if "positions" in e}
    rows = []
    for fa in sorted(seqs_dir.glob("*.fa")):
        entry = by_stem.get(fa.stem)
        if not entry:
            continue
        chain_order = entry.get("chain_order") or []
        seg = chain_order.index(binder_chain) if binder_chain in chain_order else -1
        # biased resnums -> index within the binder segment
        idx_map = entry.get("binder_index", {})
        wanted = [int("".join(c for c in p if c.isdigit())) for p in entry["positions"]]
        name = None
        first = True
        for line in open(fa):
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            elif line and name:
                parts = line.split(":")
                bseq = parts[seg] if 0 <= seg < len(parts) else line
                at = "".join(bseq[idx_map[str(p)]]
                             if str(p) in idx_map and idx_map[str(p)] < len(bseq) else "?"
                             for p in wanted)
                rows.append({
                    "source": fa.stem, "name": name.split(",")[0],
                    "is_input": first, "binder_sequence": bseq, "length": len(bseq),
                    "biased_positions": ",".join(f"{p}" for p in wanted),
                    "residues_at_biased": at,
                    "n_his_at_biased": at.count("H"),
                    "n_his_binder": bseq.count("H"),
                })
                name = None
                first = False
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Bias histidine into a binder's interface with LigandMPNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output must then be re-folded and scored with ph_scoring.py.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--complex", help="one complex PDB")
    src.add_argument("--complex_dir", help="directory of complex PDBs")
    ap.add_argument("--glob", default="*.pdb", help="pattern within --complex_dir")
    ap.add_argument("--binder_chain", default="B")
    ap.add_argument("--target_chain", default="A")
    ap.add_argument("--site", default=None,
                    help='restrict to interface positions facing this epitope, '
                         'e.g. "A:384,406,409,411,434". Must be in the numbering of the '
                         'INPUT STRUCTURES, not the original target — see --site_offset.')
    ap.add_argument("--site_offset", type=int, default=0,
                    help="added to every --site residue number before matching. Predicted "
                         "complexes are renumbered from 1 by their producer, so a site in "
                         "the original target's frame needs an offset: for a crop starting "
                         "at residue N, pass -(N-1). Example: domain III crop starts at 311, "
                         "so --site_offset -310 turns 384 into 74. Zero match is a hard error, "
                         "never a silent widening to the whole interface.")
    ap.add_argument("--his_bias", type=float, default=2.0,
                    help="logit bias added to His at interface positions (default 2.0). "
                         "Higher = more His but worse packing; 1.5-3.0 is the useful band.")
    ap.add_argument("--interface_dist", type=float, default=IFACE_CUT)
    ap.add_argument("--redesign", choices=["interface", "all"], default="interface",
                    help="'interface' keeps the rest of the binder fixed (default, "
                         "preserves the validated fold); 'all' lets MPNN repack "
                         "everything to accommodate the histidines")
    ap.add_argument("--n_seqs", type=int, default=8, help="sequences per backbone")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./mpnn_his_redesign")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU. LigandMPNN grabs the GPU whenever one is visible and "
                         "offers no flag to stop it; this hides the card instead. Use it to "
                         "run a His redesign while a design campaign owns the GPU.")
    ap.add_argument("--no_pack", action="store_true",
                    help="skip side-chain packing (faster, but ph_scoring.py then needs "
                         "re-folded structures before it can score anything)")
    ap.add_argument("--dry_run", action="store_true",
                    help="report what would be redesigned and print the command; run nothing")
    args = ap.parse_args()

    complexes = ([Path(args.complex)] if args.complex
                 else sorted(Path(args.complex_dir).glob(args.glob)))
    complexes = [p for p in complexes if p.is_file()]
    if not complexes:
        log("ERROR: no input structures found")
        return 1

    site_chain, site_nums = parse_site(args.site)
    if site_nums and args.site_offset:
        site_nums = {n + args.site_offset for n in site_nums}
        log(f"  --site_offset {args.site_offset:+d} applied: site is now "
            f"{sorted(site_nums)[:6]}{'...' if len(site_nums) > 6 else ''}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_inputs"
    work.mkdir(exist_ok=True)

    log(f"mpnn_redesign: {len(complexes)} complex(es), binder chain {args.binder_chain}, "
        f"His bias {args.his_bias}, scope {args.redesign}"
        + (f", site {args.site}" if args.site else ""))

    pdb_list, bias_path, redes_path, report = build_inputs(
        complexes, args.binder_chain, args.target_chain, args.his_bias,
        args.interface_dist, site_chain, site_nums, args.redesign, work)

    for e in report:
        if "skipped" in e:
            log(f"  SKIP {Path(e['pdb']).name}: {e['skipped']}")
        else:
            log(f"  {Path(e['pdb']).name}: {e['n_interface']} interface, "
                f"{e['n_biased']} biased"
                + (f", already His: {','.join(e['already_his'])}" if e["already_his"] else ""))
    (out_dir / "redesign_report.json").write_text(json.dumps(report, indent=1))

    if pdb_list is None:
        log("ERROR: nothing to redesign")
        return 1

    rc, _ = run_mpnn(pdb_list, bias_path, redes_path, args.binder_chain, out_dir,
                     args.n_seqs, args.temperature, args.seed, args.dry_run,
                     pack=not args.no_pack, cpu=args.cpu)
    if args.dry_run:
        log("\n  dry run — LigandMPNN was not executed")
        return 0
    if rc != 0:
        return rc

    rows = collect(out_dir, report, args.binder_chain)
    if rows:
        import csv
        out_csv = out_dir / "variants.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        log(f"\n  {len(rows)} sequence variants -> {out_csv}")
    else:
        log("\n  WARNING: LigandMPNN exited 0 but no sequences were parsed")

    packed = sorted((out_dir / "packed").glob("*.pdb"))
    if packed:
        log(f"  {len(packed)} packed full-atom models -> {out_dir/'packed'}")
        log(f"  triage now:  python ph_scoring.py --results_dir {out_dir} "
            f"--glob 'packed/*.pdb' --binder_chain {args.binder_chain} "
            f"--target_chain {args.target_chain} --skip_propka")

    log("\n  NEXT: these are UNVALIDATED sequences.")
    log("   1. re-fold each variant (Boltz-2) and check RMSD to its design backbone")
    log("   2. score the re-folded complexes with ph_scoring.py")
    log("   MPNN is pKa-blind — it placed histidines where the backbone allows them,")
    log("   not where protonation couples to binding. Step 2 is what tells them apart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
