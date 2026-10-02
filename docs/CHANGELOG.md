# Portfolio packaging — 2026-10-02

- Created an allowlisted working copy without `.env`, original Git history, archives, virtual environments, raw dataset images, large run directories, or the original PDF.
- Replaced the stale root README with distinct offline-demo, optional image-pipeline and historical-evidence instructions.
- Updated the built-in demo fixture from the obsolete `category/series` layout to the current `group/category` schema, including explicit groups/categories.
- Removed the demo's dependency on environment/provider configuration; added `--dsl` and `--output-dir`, and labeled reference images as non-parsed inputs.
- Initialized optional routes for the label-routing branch to prevent an unbound-variable failure.
- Re-enabled the already-implemented arithmetic solver in the registry; the source copy had it commented out. This changes current blackboard behavior and is not attributed to historical results.
- Allowed the aggregation utility to omit image/question text when scoring minimized records.
- Added a strict evidence verifier that rejects invalid paired evidence instead of silently dropping it.
- Added offline behavior/integrity tests and GitHub Actions configuration. Configured CI is not a claim that hosted runs have passed.
- Corrected the configuration comment: bounded reasoning-only repair is present in the CrewAI route.
- Kept historical scoring semantics unchanged and documented their numeric-normalization limitations.

The source manifest hashes describe files before these edits, not the packaged files. Historical model inference was not rerun.
