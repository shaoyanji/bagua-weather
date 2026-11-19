# Instructions for Claude Code

Read `README.md` first; it is the spec. Build **v0.1.0** in milestone order (M1 → M4).

## Rules
- Never fabricate data or results. All numbers in the README "Results" section must come from real runs.
- DWD column names, units and timestamp conventions in the README were written from memory. **Verify them against the DWD description files in the actual downloaded data before relying on them**, and fix the README table if wrong.
- Discover DWD file names by listing the directory; never hard-code zip names.
- No network access in tests. Use small fixture files in `tests/fixtures/`.
- Keep the existing tests passing (`pytest`), add tests for every new function, run `ruff check .`.
- Mask night hours (see README section 4) before any evaluation. Never evaluate on night hours.
- Chronological splits only; never shuffle time series.
- Keep dependencies minimal. `hmmlearn`/`scikit-learn` are acceptable only for M4, as optional extras.
- Commit after each milestone with a clear message. Do not commit `data/raw/` or generated images.

## Definition of done per milestone
See README section 9. When finished, update the README status line and the Results section, and list known limitations honestly.
