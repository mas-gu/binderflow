"""Application configuration and constants."""

from PyQt6.QtCore import QSettings

TOOL_COLORS = {
    "rfdiffusion": "#E53935",
    "rfdiffusion3": "#00BCD4",
    "boltzgen": "#43A047",
    "bindcraft": "#1E88E5",
    "pxdesign": "#FB8C00",
    "proteina": "#8E24AA",
    "proteina_complexa": "#00897B",
}

# Map design_id prefix to tool name
TOOL_PREFIXES = {
    "rfd_": "rfdiffusion",
    "rfdiffusion3_": "rfdiffusion3",
    "rfdiffusion_": "rfdiffusion",
    "bg_": "boltzgen",
    "boltzgen_": "boltzgen",
    "bc_": "bindcraft",
    "bindcraft_": "bindcraft",
    "px_": "pxdesign",
    "pxdesign_": "pxdesign",
    "prot_": "proteina",
    "proteina_": "proteina",
    "pc_": "proteina_complexa",
    "proteina_complexa_": "proteina_complexa",
}

# Key columns to show by default in rankings table
DEFAULT_COLUMNS = [
    "rank", "design_id", "tool", "binder_length", "combined_score", "boltz_iptm",
    "boltz_binder_plddt", "esmfold_plddt", "rosetta_dG", "rosetta_sc",
    "boltz_site_mean_pae", "site_interface_fraction", "site_centroid_dist_CA",
    "refolding_rmsd", "pDockQ", "tier",
    "binder_helix_frac", "binder_sheet_frac", "interface_KE_fraction",
]

# Columns with score gradients (higher is better unless inverted)
SCORE_COLUMNS_HIGHER_BETTER = {
    "combined_score", "boltz_iptm", "boltz_ptm", "boltz_complex_plddt",
    "boltz_binder_plddt", "esmfold_plddt", "bc_i_ptm", "bg_design_to_target_iptm",
    "px_iptm", "px_plddt", "pc_iptm", "pc_plddt", "pc_sc",
    "rosetta_sc", "rosetta_hbonds", "rosetta_dsasa", "rosetta_packstat",
    "binder_helix_frac", "binder_sheet_frac",
    "site_interface_fraction", "site_cos_angle", "pDockQ",
    "netsolp_solubility",
}

SCORE_COLUMNS_LOWER_BETTER = {
    "rosetta_dG", "boltz_site_mean_pae", "boltz_mean_interface_pae",
    "rosetta_bunsats", "refolding_rmsd", "site_centroid_dist_CA",
    "site_centroid_dist_heavy", "interface_KE_fraction", "rosetta_sap",
}

# Scatter plot presets
SCATTER_PRESETS = {
    # Binding quality
    "iPTM vs Rosetta dG": ("boltz_iptm", "rosetta_dG"),
    "iPTM vs Site PAE": ("boltz_iptm", "boltz_site_mean_pae"),
    "iPTM vs Interface PAE": ("boltz_iptm", "boltz_mean_interface_pae"),
    "iPTM vs Binder pLDDT": ("boltz_iptm", "boltz_binder_plddt"),
    # Structure quality
    "ESMFold pLDDT vs Combined Score": ("esmfold_plddt", "combined_score"),
    "Combined Score vs Rosetta dG": ("combined_score", "rosetta_dG"),
    "Structure Confidence x Docking (pLDDT vs iPTM)": ("boltz_binder_plddt", "boltz_iptm"),
    # Site specificity
    "iPTM vs SIF": ("boltz_iptm", "site_interface_fraction"),
    "iPTM vs Refolding RMSD": ("boltz_iptm", "refolding_rmsd"),
    "SIF vs Centroid Distance": ("site_interface_fraction", "site_centroid_dist_CA"),
    "SIF vs Cos Angle": ("site_interface_fraction", "site_cos_angle"),
    # Developability
    "Score vs Length": ("binder_length", "combined_score"),
    "Helix Fraction vs iPTM": ("binder_helix_frac", "boltz_iptm"),
    "Interface KE vs iPTM": ("interface_KE_fraction", "boltz_iptm"),
    "Shape Complementarity vs iPTM": ("rosetta_sc", "boltz_iptm"),
    "SAP vs iPTM": ("rosetta_sap", "boltz_iptm"),
    "Solubility vs iPTM": ("netsolp_solubility", "boltz_iptm"),
}


def get_settings():
    return QSettings("BinderBrowser", "BinderBrowser")
