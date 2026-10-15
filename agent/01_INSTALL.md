# Installing the pipeline

**On this machine everything except `unidock` is already installed** — check
`00_CAPABILITIES.md` before installing anything. This page is for a fresh machine, or for
repairing one broken tool.

## Step 0 — configure paths (do this first, it drives everything)

A fresh clone has **no `config.yaml`** — it is gitignored, because it is machine-specific. Every
tool path in the pipeline resolves through it.

```bash
# Run from the repo root — the clone IS this directory; there is no extra
# binders_pipeline_env/ level inside it.
cp config.example.yaml config.yaml
$EDITOR config.yaml          # set software_dir and weights_dir to YOUR paths
```

Resolution order, highest wins:

| Priority | Source |
|---|---|
| 1 | environment: `BINDER_SOFTWARE_DIR`, `BINDER_WEIGHTS_DIR` |
| 2 | `config.yaml` — `software_dir`, `weights_dir`, and the `tools:` / `conda_envs:` maps |
| 3 | defaults: `~/software`, `~/weights` |

Everything else is written as `${software_dir}/...` / `${weights_dir}/...` and expands from those
two. **Do not hardcode absolute paths** into the scripts — set them here once.

Check what you configured actually resolves:

```bash
python -c "
import sys; sys.path.insert(0,'binders_pipeline_env')
from config_loader import cfg
print('software_dir:', cfg.software_dir)
print('weights_dir :', cfg.weights_dir)
print('boltz env   :', cfg.conda_env('boltz'))"
```

## Layout

```
${software_dir}/        tool repos + conda envs
${software_dir}/envs/   per-tool conda envs
${weights_dir}/         model weights (large — keep OFF the home partition)
         this repo
    binders_pipeline_env/              the live pipeline (config.yaml lives here)
```

Paths are **not** hardcoded in the scripts. Everything resolves through
`config.yaml` at the repo root:

```yaml
software_dir: ${software_dir}
weights_dir:  ${weights_dir}

tools:
  bindcraft2_dir:     ${software_dir}/BindCraft2
  bindcraft2_weights: ${weights_dir}/bindcraft2
  ...
conda_envs:
  boltz:      $(conda info --base)/envs/boltz
  bindcraft2: ${software_dir}/envs/bindcraft2
  ...
```

To move the install, change `software_dir`/`weights_dir` — do not add hardcoded paths or symlinks.

## Order matters

1. **`boltz` host env first.** Both pipeline scripts run in it, and it is the hardest to get right.
2. **`bindcraft` (v1) second** — it carries PyRosetta + gemmi + DAlphaBall, which **every** tool's
   Rosetta scoring depends on. Skipping it disables interface energetics pipeline-wide.
3. Then any design tools you want.

### `boltz` host env

Needs: `boltz` (which hard-pins `gemmi==0.6.5`), numpy < 2.0, plus `plip` and `openbabel` for
interaction analysis.

```bash
conda create -n boltz python=3.11 -y
conda activate boltz
pip install boltz
pip install plip            # pulls openbabel + lxml
python -c "import boltz, gemmi, numpy; from plip.structure.preparation import PDBComplex; print('ok')"
```

> **Do not upgrade gemmi in this env.** `boltz` requires exactly 0.6.5. That version cannot parse
> some mmCIF column orders (notably BindCraft2's), so the pipeline carries its own
> `cif_to_pdb_atoms()` fallback. Upgrading to fix that would break Boltz-2, which is the validation
> backbone.

Install order is order-sensitive on CUDA 12.8 boxes: cu128 wheels, then the cuequivariance trio,
then `numpy<2.0` **last** (a pandas reinstall pulls numpy 2.x back in).

### `bindcraft` v1 env — load-bearing, install even if you never use BindCraft v1

Python **3.10** (PyRosetta constraint), with PyRosetta, gemmi and DAlphaBall. Verify:

```bash
/path/to/envs/BindCraft/bin/python -c "import pyrosetta, gemmi; print('ok')"
```

If `import gemmi` fails here, Rosetta scores 0 of N designs in every run — that is the classic
symptom.

### BindCraft2

Separate env, **Python ≥ 3.12** — it cannot share the v1 env.

```bash
git clone --depth 1 https://github.com/PacesaLab/BindCraft2.git ${software_dir}/BindCraft2
conda create -p ${software_dir}/envs/bindcraft2 python=3.12 pip -y

cd ${software_dir}/BindCraft2
export BINDCRAFT_PYTHON=${software_dir}/envs/bindcraft2/bin/python
export BINDCRAFT_WEIGHTS=${weights_dir}/bindcraft2   # keeps 5.3 GB off /home
bash install.sh cuda13          # or cuda12 for Volta/Pascal/Maxwell
```

> **Never set `BINDCRAFT_AF2_PARAMS`.** BC2 returns it verbatim without checking the directory holds
> `params_*.npz` and without downloading — pointing it at an empty dir turns `fetch-weights` into a
> silent no-op, and the install "succeeds" with no weights.

The installer picks CUDA wheels from the driver. Verify:

```bash
export BINDCRAFT_WEIGHTS=${weights_dir}/bindcraft2
${software_dir}/envs/bindcraft2/bin/python -m bindcraft.selfcheck cuda13   # want rc=0
${software_dir}/envs/bindcraft2/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Then add to `config.yaml`:
```yaml
tools:
  bindcraft2_dir:     ${software_dir}/BindCraft2
  bindcraft2_weights: ${weights_dir}/bindcraft2
conda_envs:
  bindcraft2: ${software_dir}/envs/bindcraft2
```

### Other tools

Each has per-tool install notes at `{TOOL}/INSTALL_{TOOL}.md`. Non-obvious environment requirements
the pipeline sets for you but a manual run must replicate:

| Tool | Requirement |
|---|---|
| BoltzGen | `MKL_THREADING_LAYER=GNU`, `PYTHONNOUSERSITE=1` |
| BindCraft v1 | env name is capitalised: `BindCraft` |
| BindCraft2 | `BINDCRAFT_WEIGHTS`, `PYTHONNOUSERSITE=1` |
| Proteina Complexa | UV venv, not conda; `source env.sh`; `AF2_DIR` points at BindCraft's params |
| Proteina | always uses `cuda:0` internally — restrict with `CUDA_VISIBLE_DEVICES` |
| RFdiffusion3 | Python ≥3.12, <3.13 (3.11 fails) |
| PXDesign | auto-crops targets > 300 residues |

### `unidock` — currently missing here

GPU docking for the molecule pipeline only. Without it `generate_molecules.py` falls back to
12-worker CPU Vina — slower but correct. Binder design does not use it.

## Verifying an install — `--self_test` is the gate

**Run this after every install step, and before trusting an unfamiliar machine.** It does not check
that directories exist — it imports each module, constructs the objects, and sizes the weights.

```bash
python binders_pipeline_env/generate_binders.py --self_test
echo "exit=$?"     # 0 = healthy, 1 = something is broken
```

It prints one line per check across four groups — host env, per-tool envs, weights, molecule
pipeline — and a verdict. `FAIL` lines name the problem; `warn` lines are optional components.

Every real defect found on the development box was **invisible to a directory check**: a missing
`vina` in the host env that silently zeroed 40% of every molecule score, a `unidock` env configured
but never installed, an `openfold3` checkpoint that needed a different package version. A tool whose
directory exists can still be broken; `--self_test` is what tells you.

Then confirm what the pipeline believes it can run:

```bash
python binders_pipeline_env/generate_binders.py --capabilities --deep
```

## Other verification

```bash
cd <repo>            # the binderflow checkout

# 1. every declared env exists
python -c "
import yaml, os
c=yaml.safe_load(open('config.yaml'))
e=lambda v: str(v).replace('\${software_dir}',c['software_dir']).replace('\${weights_dir}',c['weights_dir']).split('#')[0].strip()
for k,v in sorted(c['conda_envs'].items()):
    print(('OK  ' if os.path.isdir(e(v)) else 'MISS'), k, e(v))"

# 2. the pipeline imports and sees its tools
python -c "
import sys; sys.path.insert(0,'binders_pipeline_env')
import generate_binders as g; print(g.ALL_TOOLS)"

# 3. end-to-end dry run — validates args and prints commands, no GPU
python binders_pipeline_env/generate_binders.py \
    --target <a real pdb> --site "A:10-15" --length 60-80 \
    --tools <tool> --mode test --out_dir /tmp/install_check --dry_run
echo "exit=$?"     # want 0
```

The pipeline also preflights every requested tool's env at startup and exits 1 if none can run.

## A real smoke test

The cheapest true end-to-end check is one BindCraft2 trajectory against its shipped PDL1 target:

```bash
export BINDCRAFT_WEIGHTS=${weights_dir}/bindcraft2
cd ${software_dir}/BindCraft2
cat > /tmp/smoke.json <<'EOF'
{"campaign_name":"smoke","target":"hPDL1","modality":"binder",
 "binder_lengths":[55,55],"number_of_final_designs":1,"max_trajectories":2,
 "screen_steps":25,"refine_steps":10,"anneal_steps":20,"harden_steps":3,"mutate_steps":5,
 "sequence_candidates":2,"save_binder_monomers":false,"project_folder":"/tmp/bc2_smoke",
 "filters":{"Unbound_Binder_pLDDT":{"threshold":null},"pTM":{"threshold":null},
            "i_pTM":{"threshold":null},"i_pAE":{"threshold":null},
            "Backbone_Clashes":{"threshold":null},"Interface_Residues":{"threshold":null},
            "Binder_RMSD":{"threshold":null}},
 "min_iptm_anneal":null,"min_iptm_harden":null,"min_iptm_mutate":null,"min_iptm_final":null,
 "min_plddt_screen":null,"min_plddt_refine":null,"min_plddt_anneal":null,
 "min_plddt_harden":null,"min_plddt_mutate":null,"min_plddt_final":null,
 "min_monomer_plddt_final":null,"min_ptm_final":null,"max_ipae_final":null}
EOF
${software_dir}/envs/bindcraft2/bin/bindcraft design /tmp/smoke.json
ls /tmp/bc2_smoke/3_Ranked/     # want !_Ranked.csv + a .cif
```

Roughly 15 minutes on this GPU. Nulling both the filters *and* the stage gates is what makes it
accept a design at all — with defaults a short smoke run accepts nothing.
