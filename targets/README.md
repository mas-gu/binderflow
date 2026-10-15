# Curated targets — verified structures with their numbering frame

**Use a target from here when one exists.** Each entry is a symlink to the canonical file plus a
`.md` recording where it came from, **what numbering frame it uses**, and known epitopes expressed
**in that frame**.

## Why this directory exists

The tree contains the same protein several times, in **different numbering frames**, under
plausible-looking paths. A naive agent asked to design against PD-L1 found four PD-L1 files across
three tool install directories, in two frames 17 residues apart:

| File | Numbering | Same sequence? |
|---|---|---|
| `software/BindCraft2/settings/target/structures/hPDL1.pdb` | **18-132** (UniProt Q9NZQ7) | yes |
| `weights/target_data_assets/bindcraft_targets/PD-L1.pdb` | **1-115** | yes |

The repo's own curated hotspots for the PD-1 interface are `54,56,66,115` — correct for the
**18-132** file. Applied to the 1-115 file they land on a completely different face, and the run
proceeds normally for hours and produces a confidently wrong result. Nothing catches this except
looking.

**The rule: residue numbers are meaningless without the file they refer to.** Numbers from a paper,
from UniProt, or from a different PDB entry are routinely offset from the file you have.

## This directory ships docs, not structures

Structure files are **not** in the repo — they are large, often licensed, and live wherever your
`software_dir` points. Each `<NAME>.md` here records where to get the file, its numbering frame,
and verified epitopes **in that frame**.

`PDL1.pdb` is shipped by **BindCraft2 itself** at
`${software_dir}/BindCraft2/settings/target/structures/hPDL1.pdb`. Symlink it once you have BC2:

```bash
ln -s "${software_dir}/BindCraft2/settings/target/structures/hPDL1.pdb" targets/PDL1.pdb
```

Then confirm the numbering matches what `PDL1.md` records — if it does not, you have a different
copy and the epitope residues will be wrong.

## How to use

```bash
python binders_pipeline_env/generate_binders.py --preview_site \
    --target targets/PDL1.pdb --site "A:54,56,66,115" --out_dir /tmp/sitecheck
```

Exit 0 means the residues are exposed and form one patch. **Always do this**, including for targets
listed here — it costs seconds and is the only check that catches a wrong frame.

## Adding a target

1. Symlink the canonical structure here (do not copy — copies drift).
2. Write `<NAME>.md` alongside it using `PDL1.md` as the template: source, numbering frame verified
   against the actual file, known epitopes **in that frame**, and any decoys elsewhere in the tree.
3. Verify every epitope you record with `--preview_site` before writing it down.

Record the numbering you **measured**, not the one you expected:

```bash
python -c "
import sys; sys.path.insert(0,'binders_pipeline_env')
from generate_binders import get_chain_info
for c,i in get_chain_info('targets/NAME.pdb').items():
    print(c, i['length'], 'residues', min(i['residues']), '-', max(i['residues']))"
```

## Contents

| Target | File | Frame | Notes |
|---|---|---|---|
| Human PD-L1 (IgV domain) | `PDL1.pdb` | 18-132 | PD-1 interface epitope verified. See `PDL1.md`. |

This registry is a starting point, not a complete catalogue. Absence from it means "not yet
verified", not "not available".
