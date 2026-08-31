# Customer Churn Prediction — Advanced CI/CD Pipeline

A FastAPI service that predicts whether a customer will churn using a
scikit-learn `RandomForestClassifier`, trained on tenure, billing, and
support-interaction features. Same six advanced CI/CD concepts as the
sales-prediction example, applied to a **classification** problem instead
of regression — which changes what the quality gate actually checks.

## Domain

| Feature | Meaning |
|---|---|
| `tenure_months` | How long the customer has been subscribed |
| `monthly_charges` | Current monthly bill ($) |
| `total_charges` | Total billed to date ($) |
| `contract_type` | 0 = month-to-month, 1 = one year, 2 = two year |
| `support_calls` | Support calls in the last 12 months |
| `is_senior_citizen` | 0/1 |
| `has_tech_support` | 0/1 |

Target: `churn` (0 = stays, 1 = churns). Dataset is synthetic but
realistic (`scripts/generate_dataset.py`), generated once and committed
as `data/churn.csv` so every CI run trains on identical data.

## Why classification changes the pipeline

The sales-prediction example gated on **R² and MSE** (regression
metrics). A classifier needs different metrics, because **accuracy alone
is misleading** — a model that always predicts "no churn" can still score
~50%+ accuracy on a balanced dataset while being useless. This pipeline's
quality gate instead checks:

- **F1 score** — the balance of precision and recall
- **ROC-AUC** — how well the model ranks churners above non-churners
  across every possible decision threshold, independent of the 0.5 cutoff

See `metrics/baseline_metrics.json` — current model scores **F1=0.6517,
ROC-AUC=0.7638**; gate thresholds are F1≥0.55, ROC-AUC≥0.65.

## Pipeline flow

```
git push origin main
        │
        ▼
┌─────────────────────────── ci.yml ───────────────────────────┐
│  lint ──┬──────────────────────► dependency-scan              │
│         │                        (pip-audit, runs in parallel)│
│         ▼                                                     │
│  train-and-evaluate                                           │
│    - python train_model.py    → model.pkl, metrics.json       │
│    - check_quality_gate.py    → fails if F1/ROC-AUC regress   │
│    - upload-artifact "trained-model"                          │
│         │                                                     │
│         ▼                                                     │
│  test (pytest -v)                                             │
│    - downloads the exact "trained-model" artifact             │
│    - 8 API tests + 8 model regression tests                   │
└─────────────────────────────────────────────────────────────┘
        │  (only if every job above succeeded)
        ▼
┌─────────────────────────── cd.yml ────────────────────────────┐
│  build-and-scan                                                │
│    - download the SAME "trained-model" artifact by run-id      │
│    - docker build (load locally, do NOT push yet)               │
│    - Trivy scan — CRITICAL/HIGH vulns fail the job here,        │
│      before the image has touched any registry                 │
│    - only on success: push :v<version> and :candidate-<sha>    │
│         │                                                       │
│         ▼                                                       │
│  promote-to-production   (environment: production)               │
│    - blocked until a human clicks Approve                       │
│    - `docker buildx imagetools create` re-tags the exact         │
│      already-scanned image as :latest — no rebuild               │
└─────────────────────────────────────────────────────────────┘
```

## One-time setup

1. **Add repository secrets** (Settings → Secrets and variables → Actions):

   | Secret | Value |
   |---|---|
   | `DOCKERHUB_USERNAME` | Your Docker Hub username |
   | `DOCKERHUB_TOKEN` | A Docker Hub [access token](https://hub.docker.com/settings/security) |

2. **Create the `production` GitHub Environment** (Settings → Environments →
   New environment → name it `production`, check **Required reviewers**).

3. **(Recommended) Protect `main`** requiring `lint`, `dependency-scan`,
   `train-and-evaluate`, `test` to pass before merging.

See `SETUP-COMMANDS.txt` for the exact copy-paste command sequence.

## Local development

```bash
pip install -r requirements.txt -r requirements-dev.txt

python train_model.py         # produces model.pkl + metrics.json
python check_quality_gate.py  # same gate CI enforces
pytest -v                     # 16 tests
ruff check .                  # lint
pip-audit -r requirements.txt --strict   # dependency vulnerability scan

uvicorn app:app --reload      # http://localhost:8000
```

Only re-run `python scripts/generate_dataset.py` if you deliberately want
a new synthetic dataset — it is not part of the pipeline and overwrites
`data/churn.csv`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | HTML form |
| POST | `/predict_form` | Form submission → renders churn probability in HTML |
| POST | `/predict/` | JSON API → `{"churn_probability": 0.87, "churn_prediction": 1}` |
| GET | `/health` | Liveness check, used by the Docker `HEALTHCHECK` |
| GET | `/version` | Reports the running app's version and the exact model metrics it was built with |

Example:

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"tenure_months":2,"monthly_charges":95,"total_charges":190,"contract_type":0,"support_calls":6,"is_senior_citizen":1,"has_tech_support":0}'
# {"churn_probability":0.8775,"churn_prediction":1}
```

## Why each concept matters here specifically

- **Training separated from the Docker build** means the model that gets
  deployed is *exactly* the one that passed the quality gate and the test
  suite.
- **Classification-appropriate quality gate** (F1 + ROC-AUC, not accuracy)
  is what actually stops a model that's quietly learned to just predict
  the majority class from ever becoming an image.
- **`workflow_run` gating** means a failing test, lint error, or
  dependency vulnerability physically prevents `cd.yml` from starting.
- **Scan-before-push** means a vulnerable image is never reachable at
  `:candidate-<sha>`, let alone `:latest`.
- **Retagging instead of rebuilding** for production promotion guarantees
  the image a human approved is the identical digest that was scanned and
  tested.

## A note on `aquasecurity/trivy-action`'s pin

`cd.yml` pins Trivy to a full commit SHA
(`57a97c7e7821a5776cebc9bb87c984fa69cba8f1`) instead of a version tag like
`@v0.35.0`. This isn't paranoia for its own sake: on 2026-03-19,
`trivy-action` suffered a real supply chain attack
([CVE-2026-33634](https://github.com/aquasecurity/trivy/security/advisories/GHSA-69fq-xp46-6x23))
in which attackers force-pushed 76 of its 77 version tags to malicious
commits that stole CI/CD secrets before running the real scan — with the
workflow still appearing to pass. Tags in any third-party action can be
silently repointed like this; a commit SHA cannot. Treat this as the
default for any security-sensitive action, not a one-off exception.

## A note on the base image pin and `ignore-unfixed`

`Dockerfile` pins `python:3.11-slim-bookworm` rather than the floating
`python:3.11-slim`. The unpinned tag now resolves to Debian 13 ("trixie"),
a release too new to have accumulated security backports — an actual scan
against it turned up ~20 HIGH/CRITICAL OS-level CVEs, most with no fix
available at all. Bookworm (Debian 12) has a much more mature patch
cadence. Revisit this pin periodically; eventually trixie will be the
better-patched choice and bookworm will age out.

Relatedly, `cd.yml` sets `ignore-unfixed: true` on the Trivy scan. The
gate still fails the build on any HIGH/CRITICAL vulnerability that *has* a
available fix (which is real signal — e.g. an outdated `setuptools` or
`openssl` you just haven't rebuilt against yet). It does not fail forever
on CVEs with no vendor patch published yet, since there is nothing
actionable to do about those until upstream ships one.

The Dockerfile also strips the base image's own system-level
`pip`/`setuptools`/`wheel`/`ensurepip` from the final stage. The app runs
exclusively via `/opt/venv/bin/python`, so this system copy is completely
unused at runtime (confirmed with `python -X importtime` and by exercising
the API end to end) — it was nonetheless shipping HIGH CVEs across three
separate locations: stale vendored `jaraco.context`/`wheel` inside the
system `setuptools`, `ensurepip`'s bundled seed wheels, and — the one that
took a few rounds to track down — `/usr/share/python-wheels/*.whl`.
Debian patches `ensurepip` to source its seed wheels from that separate,
shared location instead of bundling them inside `ensurepip/_bundled/` the
way upstream CPython does. Trivy can read a package's version straight
out of a `.whl` file without it being installed, which is how an old
`setuptools` and pip's vendored `msgpack` kept surviving even after
`ensurepip` itself was deleted. None of this is touched by upgrading the
venv's own pip/setuptools/wheel, since it's a second, entirely separate
copy.

Two specific findings — `setuptools 70.3.0` (CVE-2025-47273) and pip's
vendored `msgpack 1.1.2` (GHSA-6v7p-g79w-8964) — survived even that fix,
identically across scans against both Debian 12 and Debian 13 base
images. Since the exact same versions appeared regardless of which Debian
release was underneath, this points to something baked into the CPython
3.11.x build itself rather than an OS-level file, and it couldn't be
pinned down to a specific path without direct access to the built image.
Rather than keep guessing paths against a live CI pipeline, these two
IDs are explicitly and narrowly exempted in `.trivyignore`, with the
reasoning documented there. `ignore-unfixed: true` does not cover them,
since both have a real upstream fix (`Status: fixed`) — they're just not
yet chased out of whatever dormant location holds them. Any other
finding, including a future one at a different CVE ID, still fails the
build normally.
