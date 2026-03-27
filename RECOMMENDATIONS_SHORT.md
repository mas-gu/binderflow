# Protein Binder Design: Recommendations Summary

*Research compiled March 2026. Based on published papers, competition data, and tool documentation.*

---

## How Many Designs to Generate

| Tool | Recommended (production) | Minimum (test) | Expected AF2/Boltz pass rate | Notes |
|------|--------------------------|-----------------|------------------------------|-------|
| RFdiffusion | 500-1000 backbones, 2 seqs each via MPNN | 100 backbones | ~1-5% pass pAE<10 | Baker lab used 10,000 backbones in original paper |
| BoltzGen | 10,000-60,000 | 1,000-5,000 | Filters to ~15 final candidates | Self-contained pipeline with Boltz-2 refolding |
| BindCraft | 300-3,000 trajectories | 100 trajectories | 10-30% accepted | Easy targets: 100 trajectories. Hard targets: up to 10,000 |
| PXDesign | 200-500 | 50-100 | Filters to 8-18 per target | Uses AF2-IG + Protenix dual filtering |
| Proteina Complexa | 200-500 | 50-100 | Self-filtering (native iPTM) | Million-scale validated by Manifold Bio |
| Proteina (backbone) | 200-500 | 50-100 | Similar to RFdiffusion | Needs MPNN + structure prediction |

**Key insight**: More designs help for backbone-only tools (RFdiffusion, Proteina) where the MPNN+filter funnel discards >95%. For end-to-end tools (BindCraft, BoltzGen, PXDesign, Proteina Complexa), fewer designs at higher quality is the trend.

## Experimental Hit Rates by Tool (Published Data)

| Tool | Hit Rate Range | Targets Tested | Best Published |
|------|---------------|----------------|----------------|
| RFdiffusion + MPNN | 1-15% | 7+ | pM affinity (TrkA) |
| BoltzGen | 66-80% of targets yield nM binders | 26 targets | pM on PDGFR |
| BindCraft | 10-100% (avg 46%) | 12 targets | nM range |
| PXDesign | 17-82% (avg ~50%) | 7 targets | nM range |
| AlphaProteo | 9-88% | 7 targets | pM range (closed-source) |
| Proteina Complexa | 40-64% | 127+ targets | pM on PDGFR |
| Latent-X | 10-64% mini-binders, 91-100% macrocycles | 7 targets | pM range (closed-source) |

## Ordering for Experimental Testing

- **Order 20-50 designs** from the top-ranked candidates across all tools
- At current hit rates (10-50%), expect 5-15 binders from 50 ordered designs
- **Expression rate**: ~95% for well-designed proteins (Adaptyv competition data)
- **Binding rate of expressed designs**: ~14% in competitive settings, 30-80% from top tools
- **Use 96-well plate format**: order 48-96 designs for a comprehensive campaign

## Critical Filtering Thresholds

| Metric | Threshold | Source |
|--------|-----------|--------|
| AF2 pAE_interaction | < 10 | RFdiffusion paper, ProteinDJ |
| AF2/Protenix ipTM | > 0.8 (relax to 0.7 for hard targets) | PXDesign, BindCraft |
| pLDDT (binder) | > 80 | All tools consensus |
| Refolding RMSD | < 2.5 A | BoltzGen |
| AF3 ipSAE_min | > 0.61 | Meta-analysis (Overath et al. 2025) |
| Rosetta dG/dSASA | Combined with ipSAE | Meta-analysis |
| Interface shape complementarity | > 0.62 | Meta-analysis |

**Best single predictor**: AF3 ipSAE_min (1.4x better average precision than ipAE). Combining with Rosetta dG/dSASA further improves prediction.

**Warning**: iPTM predicts binding (binary yes/no) but does NOT predict affinity. Do not rank by iPTM for affinity optimization.

## Binding Site Diversification Strategy

1. **Define 2-3 overlapping site definitions** with different hotspot subsets (3-6 hotspots each)
2. **Run independent campaigns** per site definition -- this increases diversity of binding modes
3. **Use hydrophobic hotspot residues** preferentially (better RFdiffusion performance)
4. **Try beta-strand conditioning** for targets with exposed edge strands (9.2% vs 0.98% success rate)
5. For each site, **run pilot with 10-20 designs** before scaling to hundreds

## Multi-Tool Strategy

**Recommended approach**: Run 3-4 tools in parallel, then merge and re-rank.

1. **Generate** with RFdiffusion + BoltzGen + BindCraft + PXDesign (or Proteina Complexa)
2. **Cross-validate** with Boltz-2 for backbone-only tools; use native scores for end-to-end tools
3. **Filter** using AF3 ipSAE or combined metric (ipSAE + Rosetta dG/dSASA)
4. **Diversify** final selection: pick top candidates from each tool, not just overall top-N
5. **Merge rankings** from multiple site definitions using rerank_binders.py

**Caution on refolding bias**: AF2/Boltz refolding can inflate confidence for sequences with evolutionary homologs. Cross-tool validation partially mitigates this.

## Expression Optimization

- **Glutamate (E) content**: strong predictor of expression (AUC 0.77). Higher E% correlates with expression
- **Lysine (K) content**: also predictive (AUC 0.73)
- **Use SolubleMPNN** for surface redesign after interface-preserving filtering
- **Alpha-helical designs** express more reliably than beta-sheet designs
- De novo designed proteins typically have high thermostability and good E. coli expression

## Proposed Run Protocol (Standard Campaign)

```
# Phase 1: Generate (parallel, ~12-24h)
python generate_binders3_new.py --target target.pdb \
  --site "A:11-17,119-124" --length 60-80 \
  --tools rfdiffusion,boltzgen,bindcraft,pxdesign,proteina_complexa \
  --mode standard --device cuda:0 --out_dir ./campaign_site1/

# Phase 2: Second site definition
python generate_binders3_new.py --target target.pdb \
  --site "A:13-20,115-122" --length 60-80 \
  --tools rfdiffusion,boltzgen,pxdesign,proteina_complexa \
  --mode standard --device cuda:0 --out_dir ./campaign_site2/

# Phase 3: Merge and re-rank
python rerank_binders.py --target target.pdb \
  --site "A:11-20,115-124" \
  --results_dir ./campaign_site1,./campaign_site2 \
  --out_dir ./merged_ranking/ --rank_only

# Phase 4: Select top 48-96 for ordering
# Pick top 10-15 per tool from merged ranking
```
