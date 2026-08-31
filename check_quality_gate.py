"""
Quality gate: fails the CI job (non-zero exit) if the freshly trained
classifier regresses below the accepted baseline in
metrics/baseline_metrics.json.

Uses F1 and ROC-AUC rather than accuracy - accuracy alone can look fine
on an imbalanced dataset even when the model has effectively stopped
distinguishing the two classes.
"""
import json
import os
import sys

METRICS_PATH = "metrics.json"
BASELINE_PATH = "metrics/baseline_metrics.json"


def main():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    f1 = metrics["f1_score"]
    roc_auc = metrics["roc_auc"]
    min_f1 = baseline["min_f1_score"]
    min_auc = baseline["min_roc_auc"]

    print(f"Trained model : F1={f1:.4f}  ROC-AUC={roc_auc:.4f}")
    print(f"Required gate : F1>={min_f1}  ROC-AUC>={min_auc}")

    failed = False
    if f1 < min_f1:
        print(f"::error::Quality gate FAILED - F1 {f1:.4f} is below minimum {min_f1}")
        failed = True
    if roc_auc < min_auc:
        print(f"::error::Quality gate FAILED - ROC-AUC {roc_auc:.4f} is below minimum {min_auc}")
        failed = True

    if failed:
        sys.exit(1)

    print("Quality gate passed")

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as fh:
            fh.write(f"f1_score={f1:.4f}\n")
            fh.write(f"roc_auc={roc_auc:.4f}\n")


if __name__ == "__main__":
    main()
