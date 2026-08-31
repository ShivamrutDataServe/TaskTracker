# syntax=docker/dockerfile:1
#
# NOTE: this image does NOT train the model. model.pkl and metrics.json
# are produced by train_model.py in CI (see .github/workflows/ci.yml) and
# must already exist in the build context before `docker build` runs.
# This keeps training reproducible, cacheable, and testable independently
# of the image, and means every image is built from a model that has
# already passed the CI quality gate.
#
# Base image is pinned to -bookworm (Debian 12) rather than the floating
# python:3.11-slim tag. As of this writing, an unpinned python:3.11-slim
# resolves to Debian 13 "trixie" - a release too new to have accumulated
# security backports yet, which showed up as ~20 HIGH/CRITICAL OS-level
# CVEs with no fix available in a Trivy scan. Bookworm has a mature,
# actively-maintained patch cadence. Revisit this pin periodically as
# trixie matures.

# ---- Stage 1: build a virtualenv with runtime dependencies only ----
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Upgrade pip/setuptools/wheel explicitly - the versions baked into the
# base image are frequently stale and were flagged by Trivy (setuptools
# CVE-2025-47273, wheel CVE-2026-24049, and pip's vendored msgpack).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: minimal runtime image ----
FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Apply whatever OS security patches Debian has published for this image
# as of build time, then drop apt's package lists to keep the image lean.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Strip the base image's own system-level pip/setuptools/wheel and the
# ensurepip seed wheels it bundles for creating new venvs. The running
# app never uses the system Python's site-packages - it runs exclusively
# via /opt/venv/bin/python (see PATH above) - so this is pure unused
# attack surface. A Trivy scan flagged HIGH CVEs living entirely in this
# unused system copy across several locations:
#   - stale vendored jaraco.context/wheel inside the system setuptools
#   - /usr/local/lib/python3.11/ensurepip (upstream CPython's bundled
#     seed wheel location)
#   - /usr/share/python-wheels/*.whl - Debian patches ensurepip to source
#     its seed wheels from here instead, a separate location upstream
#     CPython doesn't use. Trivy can read package metadata directly out
#     of a .whl file without it being installed, which is how an old
#     setuptools and pip's vendored msgpack kept surviving even after
#     ensurepip itself was removed. Confirmed by directly inspecting a
#     Debian-family filesystem for every .whl file present.
# None of this is touched by upgrading the venv's own pip/setuptools/
# wheel, since it's a second, entirely separate copy.
# `|| true` because the exact file layout can shift between Python patch
# releases; this is a best-effort hardening step, not something that
# should ever block a build if a path has moved.
RUN rm -rf \
      /usr/local/lib/python3.11/ensurepip \
      /usr/local/lib/python3.11/site-packages/pip \
      /usr/local/lib/python3.11/site-packages/pip-*.dist-info \
      /usr/local/lib/python3.11/site-packages/setuptools \
      /usr/local/lib/python3.11/site-packages/setuptools-*.dist-info \
      /usr/local/lib/python3.11/site-packages/wheel \
      /usr/local/lib/python3.11/site-packages/wheel-*.dist-info \
      /usr/local/lib/python3.11/site-packages/pkg_resources \
      /usr/local/bin/pip3 /usr/local/bin/pip3.11 \
      /usr/share/python-wheels \
    || true

COPY --from=builder /opt/venv /opt/venv

# Only what's needed to run the app - no training script, no test suite,
# no dataset, no dev tooling ends up in the final image.
COPY app.py .
COPY templates ./templates
COPY model.pkl metrics.json VERSION ./

RUN addgroup --system app \
    && adduser --system --ingroup app --no-create-home app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
