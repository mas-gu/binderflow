# Troubleshooting — what failed and what to do

## Step 1: the exit code

| Exit | Meaning | Next step |
|---|---|---|
| **0** | Ran; something ranked; no tool failed | Nothing to fix |
| **1** | Rejected before compute, or nothing usable | Read **stderr** — the message names the problem |
| **2** | argparse rejected the command line | Fix the flags; `--help` is authoritative |
| **3** | Ran, but **nothing could be ranked** | `status.json` → `per_tool` and `failure_kinds` |
| **4** | **Partial success** — ranked some, ≥1 tool failed | Results are usable; check which tool died and why |

## Step 2: `status.json` → `failure_kinds`

Every tool failure is classified. `{"bindcraft2": "oom"}` means BindCraft2 ran out of GPU memory.

| kind | What it means | What to do |
|---|---|---|
| `oom` | GPU out of memory | **Not retried** — a retry fails identically. Reduce `--length`, use a smaller target, or crop it. See the size limits in `00_CAPABILITIES.md`. |
| `missing_env` | conda env absent, or a module missing inside it | Install/repair that tool's env. Not a pipeline bug. |
| `missing_weights` | A checkpoint or weights file is absent | Fetch the tool's weights; see `01_INSTALL.md`. |
| `cuda_error` | Driver/CUDA problem, not capacity | Check `nvidia-smi`. Often a stale process holding the GPU. |
| `bad_args` | The pipeline built a command the tool rejected | Likely a pipeline bug or a bad flag combination — report it with the stderr excerpt. |
| `timeout` | Exceeded its time budget | Usually a stalled tool. BindCraft can stall on difficult targets; try `--bindcraft_filters relaxed`. |
| `unknown` | Unrecognised failure | Read the stderr excerpt in `tool_results`. |

A failing tool is **retried once** after GPU cleanup, except `oom`. If the retry also fails, the
run continues with the remaining tools and exits 4.

## Common situations

### "No designs collected. Exiting." (exit 1)
Every tool failed. Check `status.json:failure_kinds`. If they are all `missing_env`, the tools
are not installed — the preflight should have caught this, so check `config.yaml`.

### Exit 3 — designs were made but none ranked
Usually one of:
- **`--reprediction` is on and Boltz-2 failed.** With reprediction, only designs with `boltz_iptm`
  are ranked. Check `validation/boltz/`.
- **Filters removed everything.** The filter log lines say `removed N/M` for each. Loosen the
  geometric filter, or check that your `--site` is right.
- **No interface score at all.** `iptm_source` will be `N/A` in `rankings.csv`.

### A downstream tool processed 0 designs but exited 0

Two causes, both silent:

1. **Wrong extension in the glob.** `top_designs/` is mostly `.cif`; a `*.pdb` glob matches only the
   copied target. Use `rank*.cif`.
2. **Wrong numbering frame.** Boltz-2 renumbers output complexes **from 1**, so a `--site` in the
   target's numbering matches nothing. For a crop starting at residue N, subtract `N-1`.

Check what you actually fed it before trusting a zero:

```bash
ls <run>/top_designs/*.cif | wc -l          # is anything there?
python -c "import gemmi,sys; st=gemmi.read_structure(sys.argv[1]); st.setup_entities()
print([(c.name, c[0].seqid.num, c[len(c)-1].seqid.num) for c in st[0]])" <one file>
```

Full table of frames: `04_OUTPUTS.md` → "Numbering traps".

### Designs are ranked but they all dock in the wrong place
Your `--site` is probably wrong, or too loose. Check `status.json:site_residues_not_in_target` —
if non-empty, part of your site does not exist in the target and was silently dropped. Then apply
`--min_site_interface_fraction 0.25` and re-rank (no GPU needed, see `03_RERANK.md`).

### Rosetta scored 0 of N designs
`validation/rosetta/errors.log` has the reason. The classic cause is `import gemmi` failing in the
BindCraft env — check it:
```bash
"$(conda info --base)/envs/BindCraft/bin/python" -c "import gemmi, pyrosetta; print('ok')"
```
This is not fatal: `combined_score` renormalises when dG is missing, so designs are not handicapped.

### PLIP produced nothing
- `--plip_top 0` disables it.
- PLIP must be importable in the `boltz` env: `python -c "from plip.structure.preparation import PDBComplex"`.
- If PLIP runs but reports **zero interactions for every design**, the structure conversion produced
  an empty PDB. The pipeline detects this and falls back to reading the mmCIF directly, logging
  `recovered N atoms via direct mmCIF read`. If you see that line, it is working as intended.

### BindCraft2 accepted nothing
BC2 gates designs **twice**: acceptance `filters` *and* per-stage `min_iptm_*`/`min_plddt_*`
thresholds that kill trajectories mid-flight. If BC2 reports *"the settings rather than the budget
are what to change"*, use `--bindcraft2_filters no_filters`, which disables both. On a genuinely
hard target also raise `--bindcraft2_max_trajectories`.

### The run seems hung
Check whether a tool is simply slow — BoltzGen dominates `standard`/`production` runtime, and its
second refold stage **emits no log lines**. Watch file mtimes instead:
```bash
ls -lt {out_dir}/boltzgen/boltzgen_raw/intermediate_designs_inverse_folded/ | head
```
If a tool really is stuck, skip it without killing the run:
```bash
touch {out_dir}/.skip_{toolname}     # noticed within 15 s; partial output is collected
```

### Host RAM OOM (not GPU)
This machine has had hugepages lock most of RAM in the past. If a job dies to the kernel OOM killer:
```bash
cat /proc/meminfo | grep -i hugepages_total   # should be 0
sudo sysctl -w vm.nr_hugepages=0              # fix if not
```

## Re-running after a failure

Re-running the same command **does not cleanly resume**:

- Validation caches (Boltz monomer, Boltz complex, Rosetta) do resume — they are keyed on
  `design_id` and skip work already done.
- **Design tools restart from scratch**, and because `design_id` is assigned by position in a
  directory listing, stale files from the dead run can shift the numbering — which mis-keys the
  cached validation onto different molecules. BoltzGen is the exception (it passes `--reuse`).

**Safest recovery: use a fresh `--out_dir`.** If you must reuse one, delete the failed tool's
subdirectory first so its numbering starts clean.

One failure is **not** recoverable by re-running: if Boltz-2 was killed between creating a
prediction directory and writing its JSON, it will skip that record forever. Delete
`{out_dir}/validation/boltz/predictions/{design_id}/` and re-run.

## Escalating

Report with: the exact command, the exit code, `status.json`, and the tail of **stderr** (not
stdout). `failure_kinds` plus the stderr excerpt in `tool_results` is usually enough to diagnose
without reproducing.
