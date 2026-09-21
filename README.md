# infcomp — US Inflation Composite (reference domain for the LCAM regime engine)

Implements the build brief (save it as `docs/BRIEF.md`) §3 end-to-end: verified series map →
data layer → transforms → redundancy diagnostic → z-score composite (+ 0–5 L/T/M cross-check)
→ pre-committed OOS validation → HSBC-style dashboard → regime-hook JSON for the fusion layer.

## Status at hand-off

| Step (§3) | State |
|---|---|
| 1. Verified series map | **Code done, NOT run.** FRED was unreachable from the build sandbox, so every ID in `config/series_map.yaml` is a *candidate*. Run `verify` first. |
| 2. Data layer | Done. Latest vintage by default; ALFRED point-in-time via `--vintage`, publication-lag shift via `--pit`. Standalone client — swap in the TYCCLES ingestion module if preferred (needs `series_info` + `observations`). |
| 3. Transforms | Done, one module. `auto` resolves from the *verified* FRED units string. |
| 4. Redundancy diagnostic | Done. Corr matrix, PCA variance share, near-duplicate pairs, PC1 loading-flip test across expanding refits. |
| 5. Composite | Done. Expanding (or rolling) z-average; contributions sum exactly to the composite; diffusion; concentration flag; 0–5 L/T/M with ±1.25 override as its own column. |
| 6. Validation | Done, **gated**. Refuses to run until `config/gate.yaml` is filled and committed. |
| 7. Visuals | Done (PNG + PDF). Streamlit tab in `app/inflation_page.py`. |
| 8. Regime hook | Done — `outputs/inflation_regime_hook.json`, domain-agnostic schema. |

## Runbook

```bash
pip install -e ".[dev,app]"
export FRED_API_KEY=...            # free key: fredaccount.stlouisfed.org
# or: cp .env.example .env && edit .env  (git-ignored; loaded automatically via load_dotenv)
python -m infcomp demo             # synthetic smoke test, watermarked — no FRED needed
python -m infcomp verify           # writes config/series_map_verified.csv
#   -> review every row; set confirmed=TRUE on the ones you accept
#   -> drop off-FRED CSVs (date,value) in data/manual/<key>.csv and confirm them too
python -m infcomp build            # composite, diagnostics, dashboard, hook
#   -> read outputs/diag_*.csv; cut near-duplicates (confirmed=FALSE); rebuild
#   -> fill + git-commit config/gate.yaml BEFORE the next line
python -m infcomp validate --pit   # add --vintage YYYY-MM-DD for ALFRED point-in-time
python -m infcomp build            # dashboard now carries SIGNAL / DESCRIPTIVE label
pytest -q
```

## Guardrails enforced in code

- **No fabricated IDs:** only `VERIFIED` + human-`confirmed` rows are fetched. Title mismatch → `MISMATCH`, excluded. Confirmation drops automatically if a series ID changes.
- **Axiom until validated:** dashboard and hook read `UNVALIDATED` until `validate` runs; the gate file's SHA-256 is stamped into the result.
- **PCA is diagnostic:** promoted only if it beats z-avg OOS by the pre-committed margin.
- **Leakage guard:** `exclude_inputs` in the gate (never predict breakevens with breakevens).
- **No single input carries it:** `concentration.carried_by_few` in the hook.
- **Model advises, human adjusts:** `config/overrides.yaml`, clamped, separate column.

## Decisions still open

1. **z-score window** (`settings.yaml: zscore.window`). Expanding history includes the 1970s–80s for CPI/PCE and will score a 3% print as "cool". A 240m rolling window anchors to the post-1990s regime. Test both through the gate.
2. **Gate values** — target, horizon, PCA margin, signal threshold.
3. **Supercore** — shipped as *CPI services ex rent of shelter* proxy (still contains energy services). True supercore needs BLS relative-importance weights.
4. **0–5 L/T/M formulas** — expanding-percentile buckets here; align with `Heatmap_indicators_v10` before comparing like-for-like.
5. **Proprietary holdouts** (ISM prices, Conference Board, Truflation) — excluded unless you supply a CSV.
6. **Off-FRED sources** (SF Fed decomposition, NY Fed SCE, Atlanta wage tracker) are file loaders; their URLs weren't verifiable here, so no scrapers were written.
