# Protein Binder Design: Detailed Research and Recommendations

*Research compiled March 2026. All citations are from published papers, preprints, and tool documentation.*

---

## Table of Contents

1. [Optimal Number of Designs Per Tool](#1-optimal-number-of-designs-per-tool)
2. [Binding Site Diversification Strategies](#2-binding-site-diversification-strategies)
3. [Tool Combinations and Cross-Validation](#3-tool-combinations-and-cross-validation)
4. [Experimental Success Rates and Filtering](#4-experimental-success-rates-and-filtering)
5. [Proposed Run Protocols](#5-proposed-run-protocols)
6. [References](#6-references)

---

## 1. Optimal Number of Designs Per Tool

### 1.1 RFdiffusion

**Original paper protocol** (Watson et al., Nature 2023): Generated ~10,000 backbone designs per target, then 2 sequences per backbone via ProteinMPNN-FastRelax (~20,000 total designs), filtered through AF2 with initial guess and target templating. The pAE_interaction < 10 threshold is the critical filter; designs not passing this threshold "are not worth ordering since they will likely not work experimentally."

**Recommended numbers**:
- Test: 100-200 backbones (your pipeline uses 20-200)
- Standard: 500-1,000 backbones
- Production: 5,000-10,000 backbones
- 2 sequences per backbone via ProteinMPNN is standard
- Expect 1-5% to pass AF2 filters (pAE < 10, pLDDT > 80)

**Hotspot recommendations**: Use 3-6 hotspot residues. Hydrophobic residues work far better than polar/charged. Only 0-20% of specified hotspots are actually provided to the model during denoising (the rest are masked), so the model will make additional contacts beyond those specified. (RFdiffusion GitHub documentation)

**Beta-strand conditioning** (Derry et al., Nature Communications 2026): Conditioning RFdiffusion for beta-pairing at edge-strand target sites yielded 9.2% of designs meeting quality metrics, versus 0.98% for hotspot-only conditioning -- a ~10x improvement. Validated on KIT, PDGFRa, ALK-2, ALK-3, FCRL5, NRP1, and alpha-CTX with pM to mid-nM affinities.

**Pilot runs**: The RFdiffusion documentation recommends running 10 test designs with your chosen hotspots before scaling up to validate that the parameters produce reasonable results.

### 1.2 BoltzGen

**Paper protocol** (Stark et al., bioRxiv 2025): Generated 60,000 designs per target (except TNFa: 30,000), lengths 80-120, filtered down to ~15 candidates for experimental testing.

**Filtering pipeline**:
1. Structural validity check (has coordinates)
2. Backbone RMSD filter: both filter_rmsd and filter_rmsd_design <= 2.5 A
3. Amino acid composition limits: ALA, GLY, GLU, LEU, VAL each <= 20%
4. Cysteine fraction <= 0.0
5. Inverse folding via Boltz-IF
6. Refolding validation via Boltz-2 (RMSD between design and refold)
7. Biophysical analysis: H-bonds, salt bridges, dSASA, hydrophobicity
8. Quality-diversity greedy selection for final candidates

**Key metrics**: design_iptm and design_to_target_iptm are highest-priority ranking signals.

**Recommended numbers**:
- Test: 50-1,000 (your pipeline uses 50-10,000)
- Standard: 5,000-10,000
- Production: 30,000-60,000

**Experimental validation**: 80% success rate (4/5 targets yielded nM binders), with pM hits on PDGFR. Across 26 targets spanning diverse modalities, 66% success rate for novel targets with <30% sequence identity to known structures.

### 1.3 BindCraft

**Paper protocol** (Pacesa et al., Nature 2025): Tested 212 designs across 12 targets, found 65 binders (30.7% overall, average 46% per target). Target-specific success ranged from 10% (HER2) to 60% (SpCas9).

**Recommended trajectory numbers**:
- Easy targets: ~100 trajectories sufficient
- Moderate targets: 300-1,000 trajectories
- Difficult targets: 1,000-10,000 trajectories
- Generate at least 100 designs passing all filters, then order top 5-20

**Key insight**: BindCraft is an iterative AF2-hallucination method, so each trajectory is already computationally expensive (~minutes per trajectory on GPU). The quality per trajectory is higher than RFdiffusion, so fewer total designs are needed.

**Adaptyv competition performance**: BindCraft claimed 6 of 7 de novo binder spots on the leaderboard in the EGFR competition. In Round 2, 7 of 8 de novo binders used BindCraft.

**Warning**: Do not modify default filters without understanding them. The authors caution that "not all advanced settings have been systematically tested and may reduce the success rates reported."

### 1.4 PXDesign

**Paper protocol** (ByteDance/Protenix team, bioRxiv 2025): Generated binders of 60-160 aa, filtered to 8-18 candidates per target through AF2-IG + Protenix dual filtering with structural clustering via Foldseek.

**Experimental hit rates**:
- IL-7RA: 70%
- SC2RBD: 50%
- PD-L1: 82.4%
- TrkA: 17%
- VEGF-A: 62.5%
- EGFR: 43.75%
- TNF-alpha: 0%
- Overall: 17-82% across 6 of 7 targets

**Filtering**: Protenix ipTM cutoff standard, relaxed to 0.80 for VEGF-A, SC2RBD, and TNF-alpha to maintain diversity.

**Recommended numbers**:
- Test: 20-50 (your pipeline uses 20-500)
- Standard: 200
- Production: 500

### 1.5 Proteina Complexa

**Validation** (NVIDIA + Manifold Bio, March 2026): 1 million binder designs tested against 127 targets in a single multiplexed experiment, measuring over 100 million protein-protein interactions. Identified specific binders to 68% of targets tested.

**Hit rates from additional campaigns**:
- PDGFR: 63.5% hit rate, picomolar affinities
- Kinase mini-protein binders: 40-50%
- Peptide binders: 40-50%
- First de novo carbohydrate binders ever designed

**Architecture**: Fully atomistic flow-based generation. Sequences generated directly without separate inverse folding. Combines pretrained flow model with inference-time optimization.

**Recommended numbers**:
- Test: 20-50
- Standard: 200
- Production: 500-1,000

### 1.6 ProteinMPNN / LigandMPNN

**Standard practice**: 2-8 sequences per backbone. The RFdiffusion paper used 2 sequences per backbone with FastRelax cycles. ProteinDJ benchmark used 8 sequences per backbone.

**Native sequence recovery**: ~50-55% on monomers/oligomers (ProteinMPNN), 63.3% for small-molecule-interacting residues (LigandMPNN vs 50.4% for ProteinMPNN).

**SolubleMPNN**: Use soluble weights for surface redesign. Preserve interface residues within 4 A of target, redesign remaining core and surface. This significantly improved expression in the Adaptyv EGFR competition (73% to 95% expression rate between rounds).

**Speed**: ~1-2 seconds per 100 residues on GPU.

### 1.7 Saturation and Diminishing Returns

There is no published systematic study of design saturation curves. However, several data points suggest:

- For RFdiffusion: going from 1,000 to 10,000 backbones provides meaningful diversity gains. Beyond 10,000, the benefit is unclear.
- For BoltzGen: the authors chose 60,000 as production scale, suggesting they found gains up to this level. The quality-diversity greedy selection algorithm naturally handles diminishing returns by enforcing structural diversity.
- For BindCraft: easy targets saturate at ~100 trajectories. Hard targets may need thousands.
- **General principle**: Using AF2/RoseTTAFold to assess designs increases success rates ~10x (Dauparas et al., Science 2022). This means quality filtering of fewer designs is more efficient than generating massive numbers without filtering.

### 1.8 Comparison: AlphaProteo and Latent-X (Closed-Source Reference Points)

**AlphaProteo** (DeepMind, 2024): Generated designs 50-140 aa long, filtered to 47-172 candidates per target for yeast display testing. Hit rates: 9-88% across 7 targets. BHRF1: 88%. Suggests 10-100 designs sufficient for many applications after one round of screening.

**Latent-X** (Latent Labs, 2025): Tested 30-100 designs per target. Hit rates: 91-100% for macrocycles, 10-64% for mini-binders. Generated only 100 designs per length (lengths 12-18) = 700 per target, which is "over an order of magnitude fewer" than RFpeptides.

These closed-source tools represent upper bounds on what is achievable with state-of-the-art methods.

---

## 2. Binding Site Diversification Strategies

### 2.1 Hotspot Selection and Variation

**Core principle** (Fleishman lab, JMB 2012): Hotspot-centric design treats disembodied amino acid side chains as ligands, docking them exhaustively against the target surface. From 10,000 search trajectories, only the top 1% per residue identity are used.

**RFdiffusion approach**: Hotspots can be constrained to a particular epitope or allowed to explore the entire surface. When specifying hotspots, use 3-6 residues that the binder can interact with simultaneously.

**Practical site diversification protocol**:

1. **Define the full binding surface** (all residues in the region of interest)
2. **Create 2-4 overlapping hotspot sets** of 3-6 residues each:
   - Set A: hydrophobic core residues
   - Set B: shifted by 2-3 residues along the surface
   - Set C: include some polar residues for hydrogen bonding
   - Set D: edge-strand residues if beta-sheet target (use beta-pairing conditioning)
3. **Run independent campaigns** per hotspot set
4. **Merge results** using rerank_binders.py with --results_dir

### 2.2 Automated Epitope Scanning

PXDesign (ByteDance, 2025) implemented automated epitope scanning: "three different sets of hotspots were algorithmically constructed for each target, with an epitope defined by three hotspot residues automatically identified and selected." This suggests systematic scanning is beneficial.

**Recommended approach for your pipeline**:
- For a binding surface of 15-20 residues, create 3-4 subsets of 4-6 residues
- Ensure at least 2 residues overlap between adjacent subsets
- Run 200+ designs per subset with RFdiffusion, 1,000+ with BoltzGen
- End-to-end tools (BindCraft, Proteina Complexa) handle site targeting internally

### 2.3 Varying Binder Length

The RFdiffusion documentation recommends a length range (e.g., 60-80). Testing multiple length ranges can improve diversity:
- Short binders (40-60 aa): faster computation, simpler folds, may lack binding surface
- Medium binders (60-100 aa): sweet spot for most targets
- Long binders (100-150 aa): more complex topologies, higher risk of misfolding

### 2.4 Does Site Variation Improve Diversity?

**Evidence says yes**: IL-7RA was shown to have two distinct optimal binding patches (Site 1 and Site 2). Without hotspot specification, RFdiffusion defaults to the energetically preferred site. Specifying Site 2 hotspots "completely redirected" the binder designs. Running multiple site definitions generates structurally diverse binding modes, which is valuable for:
- Hedging against target conformational changes
- Generating binders with different biological mechanisms of action
- Identifying unexpected high-affinity epitopes

---

## 3. Tool Combinations and Cross-Validation

### 3.1 Standard Pipeline: RFdiffusion + ProteinMPNN + AF2

The canonical Baker lab pipeline (Watson et al., 2023):
1. RFdiffusion generates backbones (10,000 per target)
2. ProteinMPNN assigns 2 sequences per backbone (20,000 total)
3. AF2 with initial guess filters for pAE_interaction < 10
4. Rosetta calculates binding energy (dG)
5. Top candidates ordered for experimental testing

**ProteinDJ** (2025) formalized this as a modular pipeline with options for:
- Fold design: RFdiffusion or BindCraft
- Sequence design: ProteinMPNN (vanilla/soluble/hyper) or FAMPNN
- Structure prediction: AF2 Initial Guess or Boltz-2
- Analysis: PyRosetta + Biopython

Their benchmark: 100 folds, 8 sequences per fold = 800 designs, filtered by: af2_max_rmsd_binder_bndaln <= 1, af2_max_pae_interaction <= 10, af2_min_plddt_total >= 80.

### 3.2 BoltzGen Self-Contained Pipeline

BoltzGen is unique in being fully self-contained: design + inverse folding (Boltz-IF) + refolding validation (Boltz-2) + biophysical analysis + diversity-weighted ranking. No external tools needed.

**Advantage**: Consistency -- same model family for generation and validation.
**Disadvantage**: Potential self-reinforcing bias -- the same model confirms its own designs.

### 3.3 Cross-Validation and Refolding Bias

**Critical finding** (bioRxiv, December 2025): "Limitations of the refolding pipeline for de novo protein design" showed that evolutionary information (MSAs, PLM embeddings) can strongly bias folding models towards native-like predictions, causing even poorly designed sequences to receive inflated confidence metrics. This reduces the predictive performance of refolding metrics for experimental success.

**Implications for your pipeline**:
- Using Boltz-2 to validate BoltzGen designs may inflate scores (same model family)
- AF2 validation of BindCraft designs has similar risk (BindCraft uses AF2 internally)
- **Cross-tool validation is preferred**: validate BoltzGen designs with AF2/Protenix, validate RFdiffusion designs with Boltz-2
- Your pipeline's `--reprediction` flag (Boltz-2 re-prediction for all tools) provides independent cross-validation for backbone-only tools, but may introduce Boltz-2 bias for iPTM-native tools

### 3.4 Multi-Tool Consensus Design

No published paper explicitly tests consensus design (combining outputs from multiple tools and requiring agreement). However, several approaches approximate this:

- **PXDesign** uses dual filtering (AF2-IG + Protenix) -- requiring both predictors to agree increases specificity
- **Meta-analysis** (Overath et al., 2025) used AF2, ColabFold, AF3, and Boltz-1 to re-predict all 3,766 designs, finding that AF3 ipSAE_min was the single best predictor

**Recommended multi-tool protocol**:
1. Generate designs with 3-4 tools independently
2. Use a different structure predictor for validation than the one used for generation
3. Rank by combined metrics from multiple predictors
4. Select final candidates with diversity across tools

### 3.5 Head-to-Head Tool Comparisons

**Adaptyv EGFR Competition** (2024-2025):
- 601 proteins characterized, 378 expressed (95%), 53 bound (14% of expressed)
- BindCraft: 6 of 7 de novo binder spots, ~9% hit rate for de novo designs
- RFdiffusion: most popular approach but lower hit rate than BindCraft
- Cradle (optimization): #1 overall (1.21 nM) but used existing antibody as starting point
- **Key finding**: "Landing in the top 100 with respect to in silico metrics played no detectable role in increasing binding affinity"

**PXDesign vs AlphaProteo** (ByteDance, 2025): PXDesign achieved 17-82% hit rates, described as "surpassing prior methods such as AlphaProteo" on matched targets.

**BoltzGen vs RFdiffusion** (MIT, 2025): BoltzGen achieved 66% target success rate (binders for 66% of novel targets), with integrated pipeline producing nM binders directly.

---

## 4. Experimental Success Rates and Filtering

### 4.1 Meta-Analysis: 3,766 Binders (Overath et al., bioRxiv 2025)

The largest meta-analysis of computationally designed binders to date:
- **Dataset**: 3,766 experimentally tested designs across 15 targets
- **Overall success rate**: 11.6% (436 binders)
- **Severe class imbalance**: mirrors real-world challenges

**Best computational predictors**:

| Metric | Performance | Notes |
|--------|------------|-------|
| AF3 ipSAE_min | Best single predictor | 1.4x better avg precision than ipAE |
| AF3 ipSAE_min x Rosetta dG/dSASA | Best combined metric | Threshold: < -1.5 |
| AF3 LIS x interface shape complementarity | Alternative combined | Threshold: > 0.42 |
| Input interface shape complementarity | Pre-filter | Threshold: > 0.62 |
| RMSD_binder | Pre-filter | Threshold: < 3.73 A |

**Filtering strategy recommendation**:
1. Pre-filter: interface shape complementarity > 0.62 AND RMSD_binder < 3.73 A
2. Rank by AF3 ipSAE_min
3. Select top-K candidates
4. At top-10 selection, at least one binder identified for all 15 targets

### 4.2 iPTM as a Predictor

**Binary binding prediction**: iPTM is a good binary predictor of binding (yes/no) but does NOT predict affinity (Pacesa et al., Nature 2025).

**Known issues** (PMC 2025): iPTM has mathematical limitations -- sparse gradients and failure to capture full statistical likelihood. The modified actifpTM (focusing on confident interface residues) provides higher success rate.

**BindEnergyCraft** (MIT, 2025): Replaced iPTM with pTMEnergy, a statistical energy function derived from predicted inter-residue error distributions, showing improved optimization landscape.

**Practical thresholds**:
- AF2 ipTM > 0.8: standard threshold for high-confidence binders
- AF2 ipTM > 0.7: relaxed threshold for hard targets
- pAE_interaction < 10: critical binary filter
- pLDDT > 80: structural confidence threshold

### 4.3 Expression Rates

**Adaptyv EGFR Competition data**:
- Round 1: 73% expression rate (146/201)
- Round 2: 95% expression rate (378/400)
- Improvement attributed to widespread adoption of SolubleMPNN redesign

**Expression predictors** (AUC values from Adaptyv analysis):
- Glutamate (E) content: AUC 0.77 (higher E% = better expression)
- Lysine (K) content: AUC 0.73 (higher K% = better expression)
- %identity to PDB: AUC > 0.6
- TMscore: AUC > 0.6
- Alpha-helix content: positively correlated with expression
- Rosetta solvation scores: predictive

**De novo proteins**: Generally have high thermostability and good E. coli expression yield (Dauparas et al., Science 2022). Cell-free expression systems can screen 10-100s of designs efficiently.

### 4.4 Binding Rates of Expressed Proteins

- **Adaptyv competition**: 53/378 expressed proteins bound = 14%
- **BindCraft paper**: 65/212 designs bound = 30.7% (average 46% per target)
- **PXDesign paper**: 17-82% per target (6 of 7 targets yielded binders)
- **BoltzGen paper**: 80% of targets yielded nM binders
- **AlphaProteo**: 9-88% per target
- **Proteina Complexa**: 40-64% per target, 68% of 127 targets yielded binders

### 4.5 Additional Filtering Criteria Beyond iPTM/pLDDT

**Rosetta-based metrics**:
- dG (binding free energy): lower is better
- dG/dSASA (energy per surface area): combined with ipSAE is the best predictor
- Interface shape complementarity: > 0.62 as pre-filter

**Sequence-based metrics**:
- ESM3/ESM-C log-probabilities: showed "strong correlation to binding categories" (Adaptyv)
- ESM2 pseudolikelihood: AUC 0.44 for binding (weak predictor)
- Amino acid composition: excessive ALA, GLY, LEU, VAL > 20% suggests poor design

**Structural metrics**:
- Refolding RMSD < 2.5 A (BoltzGen threshold)
- Number of interface hydrogen bonds and salt bridges
- Change in solvent-accessible surface area (dSASA) upon binding
- Surface hydrophobicity metrics

**Composition filters** (from BoltzGen):
- Each of ALA, GLY, GLU, LEU, VAL should be <= 20% of sequence
- Cysteine content should be zero or near-zero

### 4.6 How Many Designs to Order

**Published examples**:
- AlphaProteo: 47-172 candidates per target for yeast display
- BoltzGen: ~15 per target after filtering 60,000
- PXDesign: 8-18 per target after clustering
- BindCraft: top 5-20 recommended
- Latent-X: 30-100 per target
- Adaptyv competition: ~400 total (across all participants)

**Practical recommendation**: Order 20-50 designs per target from merged, multi-tool rankings. If budget allows, 96 designs (one 96-well plate) provides comprehensive coverage.

---

## 5. Proposed Run Protocols

### 5.1 Quick Screen (1 target, 1-2 days compute)

Goal: Identify if binder design is feasible for a given target.

```bash
# Single site, test mode, 3 fast tools
python generate_binders3_new.py \
  --target target.pdb \
  --site "A:11-17,119-124" \
  --length 60-80 \
  --tools rfdiffusion,boltzgen,pxdesign \
  --mode test \
  --device cuda:0 \
  --out_dir ./quick_screen/
```

Expected output: ~90 designs (20 RFdiffusion + 50 BoltzGen + 20 PXDesign), ~10-20 passing filters. Order top 10 for expression testing.

### 5.2 Standard Campaign (1 target, 1-2 weeks)

Goal: Generate diverse binders with high confidence for experimental testing.

```bash
# Phase 1: Two site definitions, standard mode, all tools
python generate_binders3_new.py \
  --target target.pdb \
  --site "A:11-17,119-124" \
  --length 60-80 \
  --tools rfdiffusion,boltzgen,bindcraft,pxdesign,proteina_complexa \
  --mode standard \
  --device cuda:0 \
  --out_dir ./campaign_site1/

# Phase 2: Shifted site
python generate_binders3_new.py \
  --target target.pdb \
  --site "A:13-20,117-126" \
  --length 60-80 \
  --tools rfdiffusion,boltzgen,pxdesign,proteina_complexa \
  --mode standard \
  --device cuda:0 \
  --out_dir ./campaign_site2/

# Phase 3: Merge
python rerank_binders.py \
  --target target.pdb \
  --site "A:11-20,117-126" \
  --results_dir ./campaign_site1,./campaign_site2 \
  --out_dir ./merged/ \
  --rank_only \
  --score_weights 0.3,0.6,0.1

# Phase 4: Select top 48 for ordering
# Pick top 8-10 per tool from merged ranking
```

### 5.3 Production Campaign (1 target, max effort)

Goal: Maximize probability of finding high-affinity binders.

```bash
# Phase 1: Three site definitions, production mode
for SITE in "A:11-17,119-124" "A:13-20,117-126" "A:15-22,115-120"; do
  python generate_binders3_new.py \
    --target target.pdb \
    --site "$SITE" \
    --length 60-80 \
    --tools rfdiffusion,boltzgen,bindcraft,pxdesign,proteina,proteina_complexa \
    --mode production \
    --device cuda:0 \
    --out_dir "./production_$(echo $SITE | tr ':,' '__')/"
done

# Phase 2: Merge all runs
python rerank_binders.py \
  --target target.pdb \
  --site "A:11-22,115-126" \
  --results_dir "./production_A__11-17_119-124,./production_A__13-20_117-126,./production_A__15-22_115-120" \
  --out_dir ./production_merged/ \
  --rank_only

# Phase 3: Re-rank with different weight schemes
python rerank_binders.py \
  --target target.pdb \
  --site "A:11-22,115-126" \
  --results_dir ./production_merged/ \
  --out_dir ./production_helix/ \
  --rank_only --ss_bias helix

# Phase 4: Select 96 designs for ordering
# Top 15 per tool, diversified by structure cluster
```

### 5.4 Recommended Compute Budget Allocation

For a standard campaign with 5 tools:

| Tool | Compute time | Designs generated | Expected passing filter | Fraction of final selections |
|------|-------------|-------------------|------------------------|------------------------------|
| RFdiffusion | 4-6h | 200 backbones x 2 seq = 400 | 10-20 | 20% |
| BoltzGen | 6-12h | 1,000-5,000 | 15-50 | 25% |
| BindCraft | 8-16h | 10 trajectories | 3-8 | 15% |
| PXDesign | 4-8h | 200 | 20-40 | 20% |
| Proteina Complexa | 4-8h | 200 | 20-40 | 20% |

---

## 6. References

### Core Tool Papers

1. **Watson et al.** (2023). "De novo design of protein structure and function with RFdiffusion." *Nature* 620, 1089-1100. -- Original RFdiffusion paper. 10,000 backbones per target, ProteinMPNN sequence design, AF2 filtering.

2. **Stark et al.** (2025). "BoltzGen: Toward Universal Binder Design." *bioRxiv* 2025.11.20.689494. -- 60,000 designs per target, 66% success on novel targets, nM-pM affinities across 26 targets.

3. **Pacesa et al.** (2025). "One-shot design of functional protein binders with BindCraft." *Nature* 646, 483-492. -- 212 designs tested, 30.7% overall hit rate, 10-100% per target.

4. **ByteDance/Protenix** (2025). "PXDesign: Fast, Modular, and Accurate De Novo Design of Protein Binders." *bioRxiv* 2025.08.15.670450. -- 17-82% hit rates, AF2-IG + Protenix dual filtering, DiT architecture.

5. **NVIDIA** (2026). "Proteina-Complexa." *ICLR 2026* (oral). -- Flow-based fully atomistic binder generation. 68% of 127 targets yielded binders. Collaboration with Manifold Bio: 1M designs tested.

6. **Dauparas et al.** (2022). "Robust deep learning-based protein sequence design using ProteinMPNN." *Science* 378, 49-56. -- Standard inverse folding tool. 2-8 sequences per backbone.

7. **Dauparas et al.** (2025). "Atomic context-conditioned protein sequence design using LigandMPNN." *Nature Methods*. -- 63.3% recovery for ligand-interacting residues.

### Experimental Validation and Analysis

8. **Overath et al.** (2025). "Predicting Experimental Success in De Novo Binder Design: A Meta-Analysis of 3,766 Experimentally Characterised Binders." *bioRxiv* 2025.08.14.670059. -- AF3 ipSAE_min is best single predictor. 11.6% overall success rate. Combined metrics improve precision.

9. **Adaptyv Bio EGFR Competition** (2025). "Crowdsourced Protein Design: Lessons From the Adaptyv EGFR Binder Competition." *bioRxiv* 2025.04.17.648362. -- 95% expression, 14% binding, glutamate/lysine content predicts expression (AUC 0.77/0.73).

10. **Adaptyv Bio** (2025). "Protein Design Competition: Has binder design been solved?" Blog post. -- "Not yet -- but the rate of progress is astounding." Expression is "essentially solved."

### Methodological Advances

11. **Derry et al.** (2026). "Improved protein binder design using beta-pairing targeted RFdiffusion." *Nature Communications*. -- 9.2% vs 0.98% success with beta-strand conditioning, pM-nM affinities.

12. **Zambaldi et al.** (2024). "De novo design of high-affinity protein binders with AlphaProteo." *arXiv* 2409.08022. -- 47-172 candidates per target, 9-88% hit rates, 3-300x better affinity than prior methods.

13. **Latent Labs** (2025). "Latent-X: An Atom-level Frontier Model for De Novo Protein Binder Design." *arXiv* 2507.19375. -- 30-100 designs per target, 91-100% macrocycle hits, 10-64% mini-binder hits.

14. **ProteinDJ** (2025). "A high-performance and modular protein design pipeline." *bioRxiv* 2025.09.24.678028. -- Modular RFdiffusion/BindCraft + MPNN + AF2/Boltz-2 pipeline.

### Refolding and Validation Concerns

15. **Anonymous** (2025). "Limitations of the refolding pipeline for de novo protein design." *bioRxiv* 2025.12.09.693122. -- Evolutionary information inflates confidence metrics, compromising designability assessment. Cross-tool validation recommended.

16. **PMC** (2025). "Res ipSAE loquunt: What's wrong with AlphaFold's ipTM score and how to fix it." -- iPTM has sparse gradients and mathematical limitations. AF3 ipSAE proposed as superior metric.

17. **MIT** (2025). "BindEnergyCraft: Casting Protein Structure Predictors as Energy-Based Models for Binder Design." *arXiv* 2505.21241. -- pTMEnergy replaces iPTM with improved optimization landscape.
