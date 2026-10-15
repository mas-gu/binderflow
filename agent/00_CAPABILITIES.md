# What this machine can actually do

**Do not trust this file — ask the machine:**

```bash
python binders_pipeline_env/generate_binders.py --capabilities   # fast: what is configured
python binders_pipeline_env/generate_binders.py --self_test      # ~1 min: what actually WORKS
```

`--capabilities` needs no other arguments, touches no GPU, exits 0, and returns JSON read from the live code
and live filesystem: available tools and why any are not, GPUs and VRAM, size limits, design modes
and their cost, BindCraft2 modalities and presets, exit codes, failure kinds. It cannot go stale.

The rest of this page is a human-readable snapshot taken 2026-09-28, for orientation only.

## Hardware — ask, do not assume

```bash
python binders_pipeline_env/generate_binders.py --capabilities | \
  python -c "import json,sys; c=json.load(sys.stdin); print(c['gpus'], c['vram_gb'],'GB')"
```

Everything below is written for a **single ~20 GB GPU**, which is what this pipeline was developed
and tuned on. Your numbers come from `--capabilities`; the size limits in the next section scale
with your actual VRAM and are computed for you in its `limits` block.

If you have **more than one GPU**, `--boltz_devices N` distributes Boltz-2 validation across them —
but it *overrides* `--device` pinning and enumerates by global index, so do not combine the two.

## Design tools

`--tools` accepts exactly these 8. All envs verified present.

| Tool | Kind | Notes |
|---|---|---|
| `rfdiffusion` | backbone-only | Needs Boltz-2 to get an iPTM. PDB targets only. |
| `rfdiffusion3` | backbone-only | Python 3.12 env. Auto `diffusion_batch_size=1` when total_N > 340. |
| `boltzgen` | iPTM-native | Handles 600-700 residue targets. Dominates `standard`/`production` runtime. |
| `bindcraft` | iPTM-native | AF2-based, v1. PDB targets only. Can stall on difficult targets. |
| `bindcraft2` | iPTM-native | Rewrite. Takes PDB, **mmCIF or FASTA**. Modality presets. See §BindCraft2. |
| `pxdesign` | iPTM-native | Auto-crops targets > 300 residues. |
| `proteina` | backbone-only | Unconditional — target-agnostic. |
| `proteina_complexa` | iPTM-native | UV venv, not conda. Auto `batch_size=1` when total_N > 340. |

"iPTM-native" tools produce their own interface score, so without `--reprediction` they skip Boltz-2
complex prediction. "Backbone-only" tools always go through Boltz-2.

## Environments

All under `${software_dir}/envs/` unless noted. Host env for both pipeline scripts is
**`boltz`** (`$(conda info --base)/envs/boltz`).

**`unidock` is MISSING.** GPU molecular docking is unavailable, so `generate_molecules.py` falls
back to CPU Vina — slower, but it works. Verify before a large screen:

```bash
python binders_pipeline_env/generate_binders.py --self_test    # covers the molecule path too
```

> Until 2026-09-28 that fallback was **also** broken — `vina` was not importable in the host env,
> `mol_scoring.py` swallowed the ImportError, and the docking term (40% of `combined_score`) was
> substituted with `0.0`. Two completed runs on disk (23,138 and 6,157 molecules) have
> `vina_score` 0% populated and `combined_score` 100% populated: **they were ranked on QED+SA
> alone and nothing recorded that.** vina/meeko are now installed, scores renormalise over the
> terms actually measured, each row carries `score_terms_missing`, and a run missing a whole term
> shouts on stderr. Re-rank those two runs before trusting their ordering.

**`bindcraft` (v1) env is load-bearing beyond BindCraft itself.** It is Python 3.10 and carries
PyRosetta + gemmi + DAlphaBall, used for **Rosetta scoring of every tool's designs**, the DSSP
fallback, and Proteina Complexa's AF2 params. Never repoint `conda_envs.bindcraft` at something
else.

## Capacity limits — what will and will not fit

`total_N = target_length + max_binder_length`. Thresholds below are for a ~20 GB card; run
`--capabilities` and read `limits` for yours.

| Guard | Threshold | Behaviour |
|---|---|---|
| Proteina Complexa | total_N > 340 | auto `batch_size=1` |
| RFdiffusion3 | total_N > 340 | auto `diffusion_batch_size=1` |
| PXDesign | target > 300 res | auto-crops (±20 residues + 10 Å shell) |
| BindCraft2 | ~total_N > 415 | logs an estimate and warns; BC2 subbatches, expect slow or OOM |
| Boltz-2 complex validation | N > 700 | may OOM — watch this stage on large targets |

BindCraft2 documents its own peak as `2.0 × (3.4 GB + 38 kB × N²)`, which is what caps this card
near N=415.

## BindCraft2 specifics

- **Modalities** (`--bindcraft2_modality`): `binder` (default), `large_binder`, `peptide`,
  `cyclic_peptide`, `VHH`, `scFv`, `Fab`, `ARP`, `homo_oligomer`, `multidomain`, `fold_switch`,
  `induced_fit`. Scaffolded formats (VHH/scFv/Fab/ARP) carry their own length and **ignore
  `--length`**.
- **Filter presets** (`--bindcraft2_filters`): `no_filters` (default), `relaxed`, `default`,
  `kras_relaxed`, `peptide`, `peptide_relaxed`.
- `no_filters` disables both BC2's acceptance filters **and** its 15 per-stage gates. That matches
  v1 semantics (accept everything, let the pipeline rank) but costs GPU: hopeless trajectories run
  to completion instead of dying early.
- BC2 needs `BINDCRAFT_WEIGHTS=${weights_dir}/bindcraft2`. The wrapper sets this.
  **Never set `BINDCRAFT_AF2_PARAMS`** — it silently disables the weight download.

## Design modes — how much compute you are committing

| Mode | Designs (all 8 tools) | Wall clock, 1 GPU |
|---|---|---|
| `test` | ~160 | **~4 hours** |
| `standard` | ~1,420 | **~37 hours** |
| `production` | ~4,290 | **~5 days** |

Ask the human before anything above `test`.

## Checking for yourself

```bash
cd <repo>            # the binderflow checkout
# authoritative tool list + modes, straight from the code
python -c "
import sys; sys.path.insert(0,'binders_pipeline_env')
import generate_binders as g
print('tools:', g.ALL_TOOLS)
print('modes:', {k: sum(v.values()) for k,v in g.DESIGN_MODES.items()})"

# which envs actually exist
python -c "
import yaml, os
c=yaml.safe_load(open('binders_pipeline_env/config.yaml'))
e=lambda v: str(v).replace('\${software_dir}',c['software_dir']).replace('\${weights_dir}',c['weights_dir']).split('#')[0].strip()
for k,v in sorted(c['conda_envs'].items()):
    print(('OK  ' if os.path.isdir(e(v)) else 'MISS'), k, e(v))"

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

The pipeline also runs its own preflight at startup: it checks every requested tool's env exists
and exits 1 if **none** of them can run, listing why.
