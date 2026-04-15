"""Results API — rankings, dashboard, structure files."""

import math
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import database
from binders_pipeline_env.binder_browser.data.loader import load_rankings

router = APIRouter(prefix="/api/results", tags=["results"])


def _compute_contacts(pocket_pdb: str, ligand_sdf: str,
                       hbond_dist: float = 3.5, contact_dist: float = 4.5,
                       salt_dist: float = 4.0, pi_dist: float = 5.5) -> list[dict]:
    """
    Compute protein-ligand contacts using BioPython distance calculations.

    Returns list of dicts with: type, protein_atom, protein_residue, protein_coords,
    ligand_atom, ligand_coords, distance
    """
    from Bio.PDB import PDBParser
    from rdkit import Chem
    import numpy as np

    # Parse protein
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pocket_pdb)
    model = structure[0]

    # Parse ligand
    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=True)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m
            break
    if mol is None:
        return []

    conf = mol.GetConformer()

    # Donor/acceptor atom definitions
    hbond_donors = {'N', 'O', 'S'}  # atoms that can donate H-bonds
    hbond_acceptors = {'N', 'O', 'S', 'F'}  # atoms that can accept H-bonds
    hydrophobic_atoms = {'C'}  # simplified
    charged_pos = {'NZ', 'NH1', 'NH2', 'ND1', 'NE2'}  # Lys, Arg, His
    charged_neg = {'OD1', 'OD2', 'OE1', 'OE2'}  # Asp, Glu
    aromatic_res = {'PHE', 'TYR', 'TRP', 'HIS'}

    contacts = []
    seen = set()  # avoid duplicate contacts per residue

    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':  # skip water/heteroatoms
                continue
            res_name = residue.get_resname()
            res_id = f"{chain.id}:{res_name}{residue.id[1]}"

            for atom in residue:
                prot_coord = atom.get_vector().get_array()
                prot_elem = atom.element.strip().upper() if atom.element else atom.name[0]
                atom_name = atom.get_name()

                for i in range(mol.GetNumAtoms()):
                    lig_atom = mol.GetAtomWithIdx(i)
                    lig_elem = lig_atom.GetSymbol()
                    lig_coord = conf.GetAtomPosition(i)
                    lig_xyz = np.array([lig_coord.x, lig_coord.y, lig_coord.z])

                    dist = float(np.linalg.norm(prot_coord - lig_xyz))

                    # H-bond detection
                    if dist <= hbond_dist:
                        is_donor_acceptor = (
                            (prot_elem in hbond_donors and lig_elem in hbond_acceptors) or
                            (prot_elem in hbond_acceptors and lig_elem in hbond_donors)
                        )
                        if is_donor_acceptor:
                            key = (res_id, 'hbond', i)
                            if key not in seen:
                                seen.add(key)
                                contacts.append({
                                    "type": "hbond",
                                    "protein_residue": res_id,
                                    "protein_atom": atom_name,
                                    "protein_coords": prot_coord.tolist(),
                                    "ligand_atom": f"{lig_elem}{i}",
                                    "ligand_coords": lig_xyz.tolist(),
                                    "distance": round(dist, 2),
                                })

                    # Salt bridge detection
                    if dist <= salt_dist:
                        prot_charged = atom_name in charged_pos or atom_name in charged_neg
                        lig_charged = lig_atom.GetFormalCharge() != 0
                        if prot_charged and lig_charged:
                            key = (res_id, 'salt_bridge', i)
                            if key not in seen:
                                seen.add(key)
                                contacts.append({
                                    "type": "salt_bridge",
                                    "protein_residue": res_id,
                                    "protein_atom": atom_name,
                                    "protein_coords": prot_coord.tolist(),
                                    "ligand_atom": f"{lig_elem}{i}",
                                    "ligand_coords": lig_xyz.tolist(),
                                    "distance": round(dist, 2),
                                })

                    # Hydrophobic contact detection
                    if dist <= contact_dist and dist > hbond_dist:
                        if prot_elem == 'C' and lig_elem == 'C':
                            # Only report one hydrophobic contact per residue
                            key = (res_id, 'hydrophobic')
                            if key not in seen:
                                seen.add(key)
                                contacts.append({
                                    "type": "hydrophobic",
                                    "protein_residue": res_id,
                                    "protein_atom": atom_name,
                                    "protein_coords": prot_coord.tolist(),
                                    "ligand_atom": f"{lig_elem}{i}",
                                    "ligand_coords": lig_xyz.tolist(),
                                    "distance": round(dist, 2),
                                })

                    # Pi-stacking (simplified: aromatic residue near aromatic ligand atom)
                    if dist <= pi_dist and res_name in aromatic_res and lig_atom.GetIsAromatic():
                        if prot_elem == 'C' and atom_name in ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'):
                            key = (res_id, 'pi_stacking')
                            if key not in seen:
                                seen.add(key)
                                contacts.append({
                                    "type": "pi_stacking",
                                    "protein_residue": res_id,
                                    "protein_atom": atom_name,
                                    "protein_coords": prot_coord.tolist(),
                                    "ligand_atom": f"{lig_elem}{i}",
                                    "ligand_coords": lig_xyz.tolist(),
                                    "distance": round(dist, 2),
                                })

    # Sort by type then distance
    contacts.sort(key=lambda c: (
        {'hbond': 0, 'salt_bridge': 1, 'pi_stacking': 2, 'hydrophobic': 3}.get(c['type'], 9),
        c['distance']
    ))
    return contacts


def _parse_rankings(csv_path: Path) -> list[dict]:
    """Load rankings CSV via binder_browser loader (adds pDockQ, tier, tool)."""
    df = load_rankings(csv_path)
    # Convert to list of dicts with JSON-safe types (NaN → None)
    rows = []
    for _, row in df.iterrows():
        clean = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                clean[k] = None
            elif isinstance(v, float) and v == int(v) and k in ("rank", "binder_length", "site_n_contacted", "site_n_total"):
                clean[k] = int(v)
            else:
                clean[k] = v
        rows.append(clean)
    return rows


@router.get("/{job_id}/rankings")
async def get_rankings(job_id: str, tool: str = None, limit: int = None, include_unranked: bool = False):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    csv_path = Path(job["out_dir"]) / "rankings.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Rankings not ready yet")

    rows = _parse_rankings(csv_path)
    # Sort: ranked first (by rank), then unranked at the end
    rows.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 9999))
    # By default, exclude unranked (filtered-out) designs
    if not include_unranked:
        rows = [r for r in rows if r.get("rank") is not None]
    if tool:
        rows = [r for r in rows if r.get("tool") == tool]
    if limit:
        rows = rows[:limit]
    return rows


@router.get("/{job_id}/rankings/csv")
async def download_rankings(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    csv_path = Path(job["out_dir"]) / "rankings.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Rankings not ready")
    return FileResponse(csv_path, filename="rankings.csv", media_type="text/csv")


@router.get("/{job_id}/dashboard")
async def get_dashboard(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    img = Path(job["out_dir"]) / "dashboard.png"
    if not img.exists():
        raise HTTPException(404, "Dashboard not ready")
    return FileResponse(img, media_type="image/png")


@router.get("/{job_id}/structures")
async def list_structures(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    top_dir = Path(job["out_dir"]) / "top_designs"
    if not top_dir.exists():
        return []

    structures = []
    # Only list top-level files (ranked structures), not per-tool subdirectories
    for f in sorted(top_dir.glob("*.cif")) + sorted(top_dir.glob("*.pdb")):
        structures.append({
            "filename": f.name,
            "path": f.name,
            "tool": "global",
            "size_kb": round(f.stat().st_size / 1024, 1),
        })
    # If no top-level files, fall back to subdirectories
    if not structures:
        for f in sorted(top_dir.rglob("*.cif")) + sorted(top_dir.rglob("*.pdb")):
            if f.name.startswith("target_"):
                continue
            rel = f.relative_to(top_dir)
            structures.append({
                "filename": f.name,
                "path": str(rel),
                "tool": rel.parts[0] if len(rel.parts) > 1 else "global",
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return structures


@router.get("/{job_id}/structure/{path:path}")
async def get_structure(job_id: str, path: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    base_dir = (Path(job["out_dir"]) / "top_designs").resolve()
    file_path = (base_dir / path).resolve()
    if not file_path.exists() or ".." in path or not str(file_path).startswith(str(base_dir)):
        raise HTTPException(404, "Structure file not found")

    media = "chemical/x-cif" if file_path.suffix == ".cif" else "chemical/x-pdb"
    return FileResponse(file_path, media_type=media)


@router.get("/{job_id}/molecule/{path:path}")
async def get_molecule_file(job_id: str, path: str):
    """Serve molecule SDF or pocket PDB files for the 3D viewer."""
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if ".." in path:
        raise HTTPException(400, "Invalid path")

    out_dir = Path(job["out_dir"]).resolve()
    # Search in top_molecules/, pocket/, and tool output dirs (for reranked runs)
    search_dirs = ["top_molecules", "pocket"]
    # Also search tool subdirs for original/docked SDFs
    for d in out_dir.iterdir():
        if d.is_dir() and d.name not in ("top_molecules", "pocket", "scoring", "library"):
            search_dirs.append(d.name)
    # Also search scoring/ for docked poses
    search_dirs.append("scoring")

    for subdir in search_dirs:
        base_dir = (out_dir / subdir).resolve()
        if not base_dir.exists():
            continue
        # Direct match
        file_path = (base_dir / path).resolve()
        if file_path.exists() and str(file_path).startswith(str(base_dir)):
            if file_path.suffix == ".sdf":
                return FileResponse(file_path, media_type="chemical/x-mdl-sdfile")
            elif file_path.suffix == ".pdb":
                return FileResponse(file_path, media_type="chemical/x-pdb")
            else:
                return FileResponse(file_path)
        # Recursive search (for nested tool output dirs)
        for match in base_dir.rglob(path):
            if match.is_file() and str(match.resolve()).startswith(str(base_dir)):
                if match.suffix == ".sdf":
                    return FileResponse(match, media_type="chemical/x-mdl-sdfile")
                elif match.suffix == ".pdb":
                    return FileResponse(match, media_type="chemical/x-pdb")
                else:
                    return FileResponse(match)

    # Last resort: look up sdf_path from rankings.csv (reranked runs with absolute paths)
    csv_path = out_dir / "rankings.csv"
    if csv_path.exists() and path.endswith(".sdf"):
        design_id = path.replace("_docked.sdf", "").replace(".sdf", "")
        import csv
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("design_id") == design_id:
                    sdf = row.get("sdf_path", "")
                    if sdf and Path(sdf).exists():
                        return FileResponse(sdf, media_type="chemical/x-mdl-sdfile")
                    # Try non-docked version
                    sdf_orig = sdf.replace("_docked.sdf", ".sdf")
                    if sdf_orig != sdf and Path(sdf_orig).exists():
                        return FileResponse(sdf_orig, media_type="chemical/x-mdl-sdfile")
                    break

    raise HTTPException(404, "File not found")


@router.get("/{job_id}/contacts/{design_id}")
async def get_molecule_contacts(job_id: str, design_id: str):
    """Compute protein-ligand contacts for a docked molecule."""
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    out_dir = Path(job["out_dir"])
    pocket_pdb = out_dir / "pocket" / "pocket.pdb"

    # Find the molecule SDF (try docked first, then original)
    # Search top_molecules/ and all tool subdirs (for reranked runs)
    mol_sdf = None
    search_dirs = [out_dir / "top_molecules"]
    for d in out_dir.iterdir():
        if d.is_dir() and d.name not in ("top_molecules", "pocket", "scoring", "library"):
            search_dirs.append(d)
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for name in [f"{design_id}_docked.sdf", f"{design_id}.sdf"]:
            matches = list(search_dir.rglob(name))
            if matches:
                mol_sdf = matches[0]
                break
        if mol_sdf:
            break

    # Fallback: look up sdf_path from rankings.csv (reranked runs)
    if not mol_sdf:
        csv_path = out_dir / "rankings.csv"
        if csv_path.exists():
            import csv
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    if row.get("design_id") == design_id:
                        sdf = row.get("sdf_path", "")
                        if sdf and Path(sdf).exists():
                            mol_sdf = Path(sdf)
                        break

    if not pocket_pdb.exists() or not mol_sdf:
        return {"contacts": [], "error": "Structure files not found"}

    try:
        contacts = _compute_contacts(str(pocket_pdb), str(mol_sdf))
        return {"contacts": contacts}
    except Exception as e:
        return {"contacts": [], "error": str(e)}


@router.get("/{job_id}/plip")
async def get_plip_summary(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    plip_dir = Path(job["out_dir"]) / "plip_analysis"
    summary = plip_dir / "PLIP_SUMMARY.txt"
    if summary.exists():
        return {"summary": summary.read_text()}

    if not plip_dir.exists():
        return {"summary": "PLIP analysis not available"}

    designs = []
    for d in sorted(plip_dir.iterdir()):
        if d.is_dir():
            designs.append(d.name)
    return {"designs": designs}


def _parse_site_string(site_str: str) -> list[tuple[str, int]]:
    """Parse site like 'A:373-374,379-384' into [(chain, resnum), ...]."""
    residues = []
    if not site_str:
        return residues
    for part in site_str.strip().split():
        m = __import__('re').match(r'^([A-Za-z]):(.+)$', part)
        if not m:
            continue
        chain, ranges = m.group(1), m.group(2)
        for rng in ranges.split(','):
            rng = rng.strip()
            dash = rng.split('-')
            if len(dash) == 2:
                for r in range(int(dash[0]), int(dash[1]) + 1):
                    residues.append((chain, r))
            elif rng.isdigit():
                residues.append((chain, int(rng)))
    return residues


@router.get("/{job_id}/site_residues")
async def get_site_residues(job_id: str):
    """Return Boltz-renumbered site residues for 3D viewer highlighting.

    Boltz CIFs renumber all chains from 1. The target is chain B (binder=A).
    We compute the offset from the original PDB numbering.
    """
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    site_str = job.get("site", "")
    if not site_str:
        return {"residues": [], "chain": "B"}

    site_residues = _parse_site_string(site_str)
    if not site_residues:
        return {"residues": [], "chain": "B"}

    # Find the target PDB to determine the offset
    out_dir = Path(job["out_dir"])
    target_file = job.get("target_file", "")

    # Try to read target PDB to get first residue number
    first_resnum = None
    try:
        import gemmi
        # Try target in top_designs first, then the original
        candidates = list(out_dir.glob("top_designs/target_*.pdb")) + [Path(target_file)]
        for pdb_path in candidates:
            if pdb_path.exists():
                st = gemmi.read_structure(str(pdb_path))
                target_chain = site_residues[0][0] if site_residues else 'A'
                for chain in st[0]:
                    if chain.name == target_chain:
                        residues_list = list(chain)
                        if residues_list:
                            first_resnum = residues_list[0].seqid.num
                        break
                if first_resnum is not None:
                    break
    except Exception:
        pass

    # Compute Boltz-renumbered residue numbers
    # Boltz renumbers from 1, so: boltz_num = original_num - first_resnum + 1
    offset = (first_resnum - 1) if first_resnum is not None else 0
    boltz_residues = [r - offset for _, r in site_residues]

    return {"residues": boltz_residues, "chain": "B", "offset": offset}
