# Human PD-L1 (CD274, UniProt Q9NZQ7) — IgV domain

| | |
|---|---|
| File | `${software_dir}/BindCraft2/settings/target/structures/hPDL1.pdb` — shipped with BindCraft2. Symlink it to `targets/PDL1.pdb`. |
| md5 | `81e36fce19a8e84b934a85647c4ac66f` |
| Chain | `A`, 115 residues |
| **Numbering** | **18–132 (UniProt Q9NZQ7 frame)** |
| Sequence start | `AFTVTVPKDLYVVEYGSNMTIECK...` (so A18, F19, T20 …) |

## Verified epitope — the PD-1 binding interface

The repo's own curated hotspots, from `BindCraft2/settings/target/hPDL1.json`
(*"Human PD-L1, with the PD-1 interface as hotspots"*):

```
--site "A:54,56,66,115"
```
`PLAUSIBLE surface epitope: 4 residues, 100% exposed, spanning 10 Å` — verified 2026-09-28.

Expanded PD-1 footprint (4ZQK/3BIK contacts), filtered to solvent-exposed residues only:

```
--site "A:54,56,66,68,113,115,121,123,124,125"
```
`PLAUSIBLE surface epitope: 10 residues, 100% exposed, spanning 20 Å, Rg 7 Å` — no warnings.

Literature contacts **dropped** because they are buried in the apo structure and would only dilute
the site-fraction filters: **F19 (rel. SASA 0.17), E58 (0.06), S117 (0.18), D122 (0.16)**.

> No PD-1:PD-L1 complex exists in this tree — `hPD1.pdb` and `hPDL1.pdb` are separate monomers — so
> the expanded set is literature-derived and then filtered by measured exposure, not computed from
> contacts. The 4-residue set is the repo's own and is the safer default.

## DECOY — do not use

`${weights_dir}/target_data_assets/bindcraft_targets/PD-L1.pdb` (if present on your box)
(md5 `c26ea0db…`, also under `proteina_complexa/assets/target_data/`)

**Identical sequence, numbered 1–115** — a **17-residue offset**. Applying the hotspots above to
that file targets a completely different face, and the run proceeds normally for hours.

**One-line sanity check:** in the correct file residue 54 is **ILE**; in the decoy it is **GLU**.
Verified 2026-09-28. Check before running:

```bash
python binders_pipeline_env/generate_binders.py --preview_site \
    --target targets/PDL1.pdb --site "A:54,56,66,115" --out_dir /tmp/sitecheck
```
