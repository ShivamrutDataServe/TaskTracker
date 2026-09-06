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

## Run locally with Docker Compose

Docker Compose runs the application and a PostgreSQL 16 instance together on
an isolated network. The application is published on port `8000`.

### Prerequisites

Install the following tools and make sure they are available on your `PATH`:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the
  Docker Engine and Docker Compose v2 enabled
- Git, to clone this repository

The Docker image does not train the model. `Dockerfile` expects `model.pkl` and
`metrics.json` to exist in the build context, so generate them once before the
first Compose build. If you are also setting up the Python development
environment, follow the [Local development](#local-development) section above.

### Build and start the stack

From the repository root:

```bash
python train_model.py
python check_quality_gate.py
docker compose up --build -d
```

The Compose stack contains:

| Service | Image | Purpose | Internal port |
|---|---|---|---|
| `app` | Built from `Dockerfile` | FastAPI model-serving API | `8000` |
| `db` | `postgres:16-alpine` | PostgreSQL database | `5432` |

Compose waits for PostgreSQL's health check before starting the application.
The database data is persisted in the named `pgdata` volume. The default local
credentials are `taskuser` / `taskpass` for the `tasktracker` database; do not
reuse these values in a shared or production environment.

### Verify the running containers

```bash
docker compose ps
docker compose logs -f app
```

In another terminal, check the application and call the JSON API:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"tenure_months":12,"monthly_charges":70,"total_charges":840,"contract_type":0,"support_calls":2,"is_senior_citizen":0,"has_tech_support":0}'
```

Open <http://localhost:8000/> for the HTML form. `/version` reports the
application version and the metrics for the model packaged in the image.

Stop the containers while retaining the database volume:

```bash
docker compose down
```

To remove the containers and the local PostgreSQL data volume as well:

```bash
docker compose down -v
```

> **Current application note:** PostgreSQL is included in Compose so the full
> application stack can be exercised locally and matches the Kubernetes
> topology. The current FastAPI implementation serves the trained model and
> does not yet read or write `DATABASE_URL`; the database container is not used
> by the prediction endpoints until persistence is implemented in the app.

## Deploy to Kubernetes

The [`k8s/`](k8s/) directory contains manifests for the application,
PostgreSQL, configuration, Secrets, Services, and an optional NGINX Ingress.
The manifests use the `default` namespace and create these main resources:

- `tasktracker-app` Deployment with two application replicas
- `tasktracker-app-service` ClusterIP Service on port `80` forwarding to the
  container's port `8000`
- `postgres-db` StatefulSet with one PostgreSQL replica and a `1Gi` persistent
  volume claim
- `db-service` ClusterIP Service on port `5432`
- `tasktracker-ingress` for the host `tasktracker.internal.example.com`

### Prerequisites

You need:

- Access to a Kubernetes cluster and a configured `kubectl` context
- A container registry if deploying to a remote cluster
- An NGINX Ingress Controller only if you intend to use `ingress.yaml`
- A storage class that can provision the StatefulSet's `1Gi` PVC

Check the selected cluster before applying anything:

```bash
kubectl config current-context
kubectl cluster-info
kubectl get nodes
```

### Build the image

The Kubernetes Deployment currently references `tasktracker:latest` with
`imagePullPolicy: IfNotPresent`.

For Docker Desktop Kubernetes, build the image in the Docker environment used
by the cluster:

```bash
python train_model.py
python check_quality_gate.py
docker build -t tasktracker:latest .
```

For a remote cluster, tag and push the image to a registry that the cluster
can access, then update `k8s/app-deployment.yaml` so
`spec.template.spec.containers[0].image` uses that fully qualified image. For
example:

```bash
docker build -t REGISTRY.example.com/TEAM/churn-prediction:VERSION .
docker push REGISTRY.example.com/TEAM/churn-prediction:VERSION
kubectl -n default set image deployment/tasktracker-app \
  app=REGISTRY.example.com/TEAM/churn-prediction:VERSION
```

Use an immutable version tag rather than `latest` for shared environments.
The registry credentials, if required, must be configured as a Kubernetes
`imagePullSecret` and referenced by the Deployment; the supplied manifests do
not create one.

### Apply the manifests

Review and replace the example Secret values before applying them. The checked-in
secrets are base64-encoded demonstration values, not encryption. For a real
cluster, create Secrets through your organization's secret-management process
and do not commit production credentials.

Apply the database resources first, followed by the application resources:

```bash
kubectl apply -f k8s/db-secret.yaml
kubectl apply -f k8s/db-configmap.yaml
kubectl apply -f k8s/db-service.yaml
kubectl apply -f k8s/db-statefulset.yaml

kubectl apply -f k8s/app-secret.yaml
kubectl apply -f k8s/app-configmap.yaml
kubectl apply -f k8s/app-service.yaml
kubectl apply -f k8s/app-deployment.yaml
```

The database and application manifests have health probes, but applying them
in this order makes the dependency relationship easier to inspect. The app's
`DATABASE_URL` points to `db-service:5432` inside the cluster.

### Verify the deployment

Wait for the database and application to become ready:

```bash
kubectl get pods -w
kubectl get statefulset postgres-db
kubectl get deployment tasktracker-app
kubectl get pvc
kubectl rollout status statefulset/postgres-db
kubectl rollout status deployment/tasktracker-app
```

Inspect service endpoints and application logs if a pod is not ready:

```bash
kubectl get services
kubectl get endpoints db-service tasktracker-app-service
kubectl logs deployment/tasktracker-app
kubectl describe pod -l app=tasktracker-app
```

For a cluster without an externally exposed load balancer, use port forwarding
to verify the API from your workstation:

```bash
kubectl port-forward service/tasktracker-app-service 8000:80
```

Then, in another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/version
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"tenure_months":12,"monthly_charges":70,"total_charges":840,"contract_type":0,"support_calls":2,"is_senior_citizen":0,"has_tech_support":0}'
```

To use the supplied Ingress, install/configure an NGINX Ingress Controller,
apply the manifest, and make `tasktracker.internal.example.com` resolve to the
Ingress controller's address (for example, with a DNS record or a local
`hosts` entry):

```bash
kubectl apply -f k8s/ingress.yaml
kubectl get ingress tasktracker-ingress
```

The Ingress manifest routes `/` to `tasktracker-app-service` and assumes the
NGINX ingress class. If your cluster uses another controller or hostname,
update `k8s/ingress.yaml` before applying it.

### Update and remove the deployment

After building and publishing a new image, update the Deployment and watch the
rolling update:

```bash
kubectl -n default set image deployment/tasktracker-app \
  app=REGISTRY.example.com/TEAM/churn-prediction:NEW_VERSION
kubectl rollout status deployment/tasktracker-app
```

Remove the application and database resources when the environment is no
longer needed:

```bash
kubectl delete -f k8s/ingress.yaml --ignore-not-found
kubectl delete -f k8s/app-deployment.yaml -f k8s/app-service.yaml \
  -f k8s/app-configmap.yaml -f k8s/app-secret.yaml
kubectl delete -f k8s/db-statefulset.yaml -f k8s/db-service.yaml \
  -f k8s/db-configmap.yaml -f k8s/db-secret.yaml
```

Deleting the StatefulSet does not necessarily delete its persistent volume
claim. Delete the claim explicitly only when its data is no longer needed:

```bash
kubectl delete pvc postgres-data-postgres-db
```

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
