# Historical ablation results

This package preserves the previously completed Caserta ablation studies. It is **historical**: all GP and MEAN6 results here use the **retired GP implementation**, not the replacement GP used by current MAIN and timing. No historical GP value is presented as a replacement-GP ablation result. The replacement-GP ablation/P0/multi-K studies remain canceled.

## Included studies

- Component/window ablation: UWSAA versus no-window-shuffle, single-window B2/B5/B10/B20, and preselected best fixed H.
- Deterministic-single comparison (distinct from K=1 rollout).
- Scenario-count/rollout and LB1 controls, retaining all original positive and negative estimates and ordinary/adjusted confidence intervals.

`data/` contains original statistical tables, condition/geometry results, accepted primary contrasts, and the available accepted raw historical endpoint records. Source row IDs, hashes, evidence statuses and solver labels are preserved. `source_inventory.json` records exact row coverage and the original payload identity; shared MAIN reference endpoints are not duplicated in the historical raw subset.

`figures/` labels GP as retired and MEAN6 as historical. These figures describe the old study and must not be combined into the current six-solver claim. Run `scripts/generate_figures.py --output-dir <directory>` to recreate them from the supplied CSVs. The original paper-facing terms are used for relocation counts and relocations per container.

The 493,920 raw historical endpoint rows are distributed as two ordered byte parts, with hashes in `data/historical_ablation_endpoints.parts.json`. To reconstruct the gzip CSV from the repository root:

```bash
python reproducibility/data/reassemble_endpoints.py --index results/historical-ablation-20261021/data/historical_ablation_endpoints.parts.json --output work/historical_ablation_endpoints.csv.gz
```

`python checks/verify_historical_ablation.py` verifies the parts and their endpoint coverage directly, without writing the reconstructed payload.

Intervals retain their original confidence levels and multiplicity adjustments. A positive point estimate is not proof that every control is beaten; intervals crossing zero and failed/negative cells remain visible. Historical timing or performance values are not substitutes for current measurements.
