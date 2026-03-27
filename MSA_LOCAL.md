# MSA Options for Boltz-2 Validation

## The Problem

Boltz-2 batch validation requires MSA (Multiple Sequence Alignment) for each design. The default uses `api.colabfold.com` (public MMseqs2 server), which is:
- Rate-limited (no formal docs, ~1 query per second practical limit)
- Slow at scale (~30-40 records/hour for 1000+ designs)
- Unreliable (intermittent timeouts reported in ColabFold GitHub #606, #664)
- Running multiple parallel jobs does NOT increase throughput

For a standard campaign (1134 designs), MSA preprocessing takes **~8-12h** — often longer than the actual Boltz-2 inference (~10h).

## Options

### Option 1: Pre-computed Target MSA + Empty Binder MSA (Recommended)

Boltz-2 YAML supports `msa` field per chain:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: TARGET_SEQ_HERE
      msa: /path/to/target_msa.a3m    # pre-computed once
  - protein:
      id: B
      sequence: BINDER_SEQ_HERE
      msa: "empty"                     # de novo = no homologs
```

**How to pre-compute target MSA:**
```bash
# Run Boltz-2 on target alone (generates MSA, saves to cache)
conda activate boltz
boltz predict target.yaml --out_dir ./target_msa/ --use_msa_server
# Find the A3M file in ./target_msa/boltz_results_target/msa/
```

**Pros:**
- MSA phase drops from hours to seconds
- Target MSA computed once, reused across all designs
- Binder MSA correctly empty (de novo sequences have no homologs anyway)

**Cons:**
- Boltz warns "predictions will be suboptimal without an MSA" for empty chains
- May slightly reduce iPTM accuracy (not validated yet)

**Implementation:** Modify `validate_boltz()` in generate_binders.py to:
1. Pre-run MSA for target sequence once
2. Reference target A3M file in each design YAML
3. Set binder MSA to "empty"

### Option 2: Local MMseqs2 Server

Run your own MSA server against locally downloaded databases.

**Requirements:**
- ~1 TB disk for databases (UniRef30 + ColabFoldDB)
- **768 GB+ RAM** to keep databases in memory (critical for speed)
- Without sufficient RAM, local search is **4-8x SLOWER** than the API

**Databases needed (MMseqs2 format, NOT the HHsuite databases we have for RFAA):**
```
uniref30_2302_db    (~150 GB)  — clustered UniProt
colabfold_envdb     (~800 GB)  — environmental sequences
```

Note: Our existing RFAA databases (UniRef30_2020_06, pdb100) are in **HHsuite format** (ffdata/ffindex) and are NOT compatible with MMseqs2 search. Separate download required.

**Setup (from ColabFold docs):**
```bash
# Download databases
DBBASE=/localdata/mmseqs_databases
mkdir -p $DBBASE
cd $DBBASE

# UniRef30 (~150 GB download)
wget https://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2302.tar.gz
tar xzf uniref30_2302.tar.gz

# ColabFold environmental DB (~800 GB download, optional but recommended)
wget https://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz
tar xzf colabfold_envdb_202108.tar.gz

# Start local server
# See: https://github.com/sokrypton/ColabFold/blob/main/MsaServer/README.md
mmseqs server $DBBASE/uniref30_2302_db $DBBASE/colabfold_envdb_202108_db \
    --port 8080 --threads 16
```

**Then point Boltz to local server:**
```bash
boltz predict input.yaml --use_msa_server --msa_server_url http://localhost:8080
```

**Our system:**
- `/localdata` has 2.7 TB free — enough for databases
- RAM: need to check, but likely ~256 GB (insufficient for in-memory databases)
- Without in-memory DBs, local search would be disk-bound and slower than API

### Option 3: colabfold_search (Local Batch Search)

Use ColabFold's local search tool (already installed) to pre-generate MSAs:

```bash
# Search against downloaded databases
/home/masgu/data/software/localcolabfold/colabfold-conda/bin/colabfold_search \
    input.fasta /path/to/mmseqs_databases output_dir \
    --db1 uniref30_2302_db --threads 16
```

This generates A3M files that can be referenced in Boltz YAML (Option 1).

**Limitation:** Still requires the MMseqs2-format databases (~1 TB download).

### Option 4: Keep Using API (Current)

Accept the speed limitation. Practical for test mode (~40 designs → 20 min).
Painful for standard mode (~1100 designs → 8-12h).

## Recommendation

**Short term:** Implement Option 1 (pre-computed target MSA + empty binder MSA). Zero infrastructure change needed. Test on a small batch to validate iPTM scores are comparable.

**Medium term:** If iPTM scores are degraded with empty MSA, download MMseqs2 databases to `/localdata` and use `colabfold_search` for batch pre-computation of all MSAs.

**Long term:** If running frequent large campaigns, set up local MMseqs2 server (requires RAM upgrade to 768 GB+).

## References

- [Boltz YAML MSA field](https://github.com/jwohlwend/boltz) — `msa: path/to/file.a3m` or `msa: "empty"`
- [ColabFold MSA Server setup](https://github.com/sokrypton/ColabFold/blob/main/MsaServer/README.md)
- [MMseqs2 local runs slower than API](https://github.com/soedinglab/MMseqs2/issues/658)
- [ColabFold API timeouts](https://github.com/sokrypton/ColabFold/issues/606)
- [Boltz local MSA generation guide](https://github.com/jwohlwend/boltz/issues/48)
- [Rowan hosted MSA server](https://www.rowansci.com/blog/msa-failures-and-our-response)
