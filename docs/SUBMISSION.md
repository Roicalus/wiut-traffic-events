# Submission checklist (per the organisers' requirements)

Steps before the final tag. Everything that can be checked automatically is
checked by `python tools/presubmit.py`.

1. Zone reference frame: `python tools/make_zone_ref.py --video samples/C3896.MP4 --frame 150`,
   check with `python tools/check_alignment.py --videos samples`.
2. Weights are in `weights/` (and committed; they are ~25 MB).
3. Local torch = 2.8.0 (as in requirements): `python -c "import torch; print(torch.__version__)"`.
4. Final run on ALL samples WITHOUT --no-risk and without WIUT_TIME_GUARD:
   `python run_submission.py --videos samples --out predictions_samples.json --team SoWeNeedAName`
5. `python evaluate.py --pred predictions_samples.json --validate-only`
6. README: fill in the team (who did what, links), remove `<team name>`.
7. Determinism: `python tools/presubmit.py --determinism samples/C3905.MP4`
8. `git add -A && git commit`, `python tools/presubmit.py` → 0 FAIL.
9. "Clean machine" check: clone the repository into a new folder, new venv,
   `pip install -r requirements.txt`, `python run_submission.py --videos samples --out /tmp/p.json`.
   Best done in a Kaggle Notebook with a T4 (the same GPU the organisers use).
10. `git tag v1.0 && git push origin main --tags`; the repository is public.
11. Submit: repository link + tag/commit hash, website link.
