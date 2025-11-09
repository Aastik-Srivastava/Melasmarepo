## MelaScan — File & Purpose Overview

This document summarizes the main files and directories in this repository, what each component does, and recommendations about what can be safely removed, archived, or must be kept for the frontend + ML melasma detection app.

Notes and assumptions
- This repo is a Django-based webapp used to upload skin images and produce melasma detection reports (see `README.md`).
- I focus on runtime/frontend files and ML integration components relevant to end-to-end operation. Remove/archival recommendations assume you have backups of any heavy model files or DB files before deletion.

---

## Top-level files
- `README.md` — Project overview, setup & usage. Keep. Useful for onboarding.
- `requirements.txt` — Python dependencies. Keep; consider pinning unspecified packages (e.g., `tensorflow` currently unpinned).
- `manage.py` — Django CLI entrypoint. Keep.
- `db.sqlite3` — Development SQLite DB. Keep for local dev, but do NOT commit large production DBs. If you are packaging or deploying, either remove this from git and add to `.gitignore` or export / migrate data and archive copy.

## Django project folder: `melascan/`
- `settings.py` — Main Django configuration. Keep. Edit for production (SECRET_KEY, DEBUG=False, ALLOWED_HOSTS). Contains media/static paths and Supabase env handling.
- `urls.py`, `wsgi.py`, `asgi.py`, `__init__.py` — Standard Django scaffolding. Keep.

## Main app: `detection/`
This folder contains the application logic for user accounts, uploads, the ML pipeline glue, and reporting.

- `__init__.py` — Keep.
- `admin.py` — Django admin registrations. Keep if you use the admin interface.
- `apps.py` — App configuration. Keep.
- `decorators.py` — Custom decorators (e.g., Supabase login-required and helpers). Keep if authentication uses Supabase.
- `forms.py` — Django forms for registration, profile, and image upload. Keep.
- `views.py` — Primary views for registration, login, logout, dashboard, profile, detect, download PDF. Critical for frontend — keep.
- `views_hybrid.py` — Views for hybrid segmentation+classification flow (separate UI). Keep if you use the hybrid pipeline frontend (`hybrid_index.html`). If you don't plan to expose the hybrid interface, this can be archived but recommended to keep (it integrates segmentation + classification endpoints).
- `hybrid_pipeline.py` — Utilities to discover/load torch segmentation/classification models, run segmentation, choose best models. Keep if you use the hybrid pipeline or any `.pt` segmentation models.
- `ml_integration.py` — Glue code that loads models (segmentation and classification), preprocesses images, and produces predictions. KEEP — this is the core ML inference layer used by `views.py`.
- `ml_models/` — Subpackage with `classification_model.py`, `segmentation_model.py` and wrappers. Keep — contains model architecture and helpers referenced by `ml_integration.py` and `hybrid_pipeline.py`.
- `models.py` — Django models: `UserProfile`, `ModelPerformance`, `MelasmaReport`. Keep — database schema for reports and model metrics.
- `report_generator.py` — PDF generation (ReportLab). Keep if report generation is used.
- `supabase_auth.py` — Integration wrappers for Supabase auth and session storage. Keep unless you move to Django-native authentication.
- `urls.py` — App routing. Keep.
- `management/commands/` — Contains utility management commands:
  - `init_models.py` — likely used to seed model metadata. Keep if you use it.
  - `test_models.py` — test helper; safe to archive if unused but OK to keep.

### Files you can consider deleting or archiving from `detection/`
- Any test/experimental scripts that are not referenced by the app or CI. For example `management/commands/test_models.py` may be archived if not needed.
- If you don't use Supabase at all, `supabase_auth.py`, forms referencing Supabase, and related decorators can be removed — but only after replacing auth flows.

## Templates: `templates/` (frontend)
- `base.html` — Main layout. Keep.
- `detection/` templates: `dashboard.html`, `detect.html`, `hybrid_index.html`, `login.html`, `profile.html`, `register.html` — Keep. These render the frontend pages and are essential.

Deletion notes: If you have duplicate or unused templates (e.g., old variants), move them to an `archive/` folder rather than deleting immediately.

## Static and media
- `static/` — CSS/JS/images used by frontend. Keep; ensure `STATICFILES_DIRS` and `collectstatic` configuration are correct for deployment.
- `media/` — Uploaded files and models. Contains subfolders:
  - `uploads/` — user uploaded images (keep or archive according to data policy)
  - `reports/` — generated PDFs
  - `models/` — heavy ML artifacts (`hybrid_best.pt`, `fcm_classifier.pkl`, ...)

Deletion recommendations for `media/models/`:
- DO NOT delete model files unless they are backed up. They are large and required for inference. If you need a smaller repo, add large model files to `.gitignore` and store models in a model registry or external storage (S3, supabase storage, Google Drive) before removing from the repo.

## Jupyter notebooks: `MLMODELS/`
- Contains many notebooks (`*.ipynb`) such as `dtreefinal.ipynb`, `FATNet.ipynb`, `hybrid.ipynb`, `ridge_melasma.ipynb`, etc.

Recommendation:
- These are development artifacts (training/experiments). They are NOT required for runtime. You can safely archive them to a separate folder (or a dedicated repo) to reduce clutter in the runtime repo. Keep them if you want reproducibility/history, but consider removing them from production branch.

## Other files / folders
- `ML_MODELS_SETUP.md`, `MLMODELS/` docs — Keep as documentation if they describe training/integration. Otherwise archive.
- `INTEGRATION_SUMMARY.md`, `SETUP_COMPLETE.md`, `QUICKSTART.md`, `SUPABASE_SETUP.md` — Keep; useful references.

## Files that are likely safe to delete or move to an archive (with caveats)
- `MLMODELS/` notebooks — move to an `archive/` or separate `research/` repository.
- Large model checkpoint files in `media/models/` — remove only if backed up externally. Better: move them out of git and add them to `.gitignore`.
- `db.sqlite3` — safe to delete if you have no dev data to preserve; otherwise export data or add to `.gitignore` and keep a backup.
- Old experimental scripts that are not referenced anywhere — move to `archive/`.

## Suggested immediate clean-up steps (safe workflow)
1. Create a branch `cleanup/archive-notebooks`.
2. Move `MLMODELS/` into `archive/MLMODELS/` or a separate repo.
3. Add `media/models/*` and `db.sqlite3` to `.gitignore` to avoid committing large/binary files.
4. If you need to remove large models from Git history, use BFG or git filter-repo (careful; repository rewrite).
5. Backup the `media/models/` files to external storage (S3, supabase storage, Google Drive) before deletion.

## Quick mapping (file → purpose) — concise
- `manage.py` — Django CLI
- `requirements.txt` — deps
- `melascan/settings.py` — app config + static/media paths
- `detection/views.py` — main frontend endpoints (login/register/dashboard/detect)
- `detection/views_hybrid.py` — endpoints for hybrid segmentation/classification UI
- `detection/ml_integration.py` — ML glue: load models, preprocess, predict
- `detection/hybrid_pipeline.py` — discovery and utilities for multiple models (.pt, .pkl)
- `detection/ml_models/` — model architecture + wrappers
- `detection/models.py` — DB models (reports, model metrics, user profile)
- `detection/report_generator.py` — create PDF reports
- `detection/supabase_auth.py` — Supabase auth wrapper
- `media/models/` — trained model checkpoints and metrics files
- `templates/detection/*.html` — frontend pages
- `static/` — css/js used by templates

## Next steps & recommendations
- Add a `.gitignore` that excludes `media/models/*`, `db.sqlite3`, `__pycache__/`, and large artifacts.
- If you prefer a smaller repo for deployment, move notebooks and training artifacts to a separate repo.
- Add a small README section explaining where heavy models are stored and how to download them for local dev.
- Consider pinning versions in `requirements.txt` (e.g., `tensorflow==2.15.0`) to ensure reproducible installs.

---

If you'd like, I can:
- Create the `.gitignore` and move notebooks into an `archive/` folder (I can make the moves in a branch), or
- Produce a short cleanup script to archive and remove large files safely, or
- Generate a smaller runtime-only bundle containing only needed files for deployment.

Tell me which of these you want next and I'll perform it (I can make changes directly in the repo).

---
Generated: automatic audit based on repository files present in the workspace.
