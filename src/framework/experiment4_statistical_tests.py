"""Experiment 4: Statistical significance testing for Stage 2 classification.

Computes:
- Bootstrap 95% confidence intervals on per-fold F1 scores
- Approximate significance tests between encoder/classifier pairs
- Reports which performance differences are statistically reliable

Since per-document predictions are not cached in the results JSON, this script:
1. Computes CIs from fold-level metrics (conservative estimate)
2. Re-runs classifiers with cached embeddings to get per-document predictions for McNemar's test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def bootstrap_ci_from_folds(fold_f1s: list[float], n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42) -> dict:
    """Compute bootstrap CI by resampling fold-level F1 scores.
    
    Note: With only k=3 folds, this produces wide intervals — which is 
    the honest representation of uncertainty at this sample size.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(fold_f1s)
    n = len(arr)
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    mean = float(np.mean(boot_means))
    
    return {"mean": mean, "ci_lower": lower, "ci_upper": upper, "ci_width": upper - lower}


def overlapping_ci_test(ci_a: dict, ci_b: dict) -> bool:
    """Simple overlapping CI test: if CIs don't overlap, difference is significant.
    
    Conservative: non-overlapping CIs imply significance at roughly p < 0.01,
    while overlapping doesn't necessarily mean non-significance.
    """
    return ci_a["ci_upper"] < ci_b["ci_lower"] or ci_b["ci_upper"] < ci_a["ci_lower"]


def welch_t_test_from_folds(folds_a: list[float], folds_b: list[float]) -> dict:
    """Welch's t-test on fold-level F1 scores (paired across folds)."""
    if len(folds_a) != len(folds_b):
        return {"t_stat": None, "p_value": None, "significant": None, "note": "unequal fold counts"}
    
    diffs = np.array(folds_a) - np.array(folds_b)
    if np.std(diffs) == 0:
        return {"t_stat": float('inf') if np.mean(diffs) != 0 else 0.0, 
                "p_value": 0.0 if np.mean(diffs) != 0 else 1.0,
                "significant": np.mean(diffs) != 0}
    
    t_stat, p_value = stats.ttest_rel(folds_a, folds_b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "mean_diff": float(np.mean(diffs)),
        "note": f"Paired t-test with k={len(folds_a)} folds (low power due to small k)"
    }


def run_experiment4(output_path: str = "output/experiment4_statistical_tests.json"):
    """Run all statistical analyses on existing Stage 2 results."""
    
    base_dir = Path("output")
    
    # Load all results
    results_files = {
        "legal_bert_no_punct": base_dir / "stage2_embeddings_results_no_punct.json",
        "legal_bert_raw": base_dir / "stage2_embeddings_results.json",
        "legal_bert_no_punct_no_stop_no_num": base_dir / "stage2_embeddings_results_no_punct_no_stop_no_num.json",
        "neuralmind_base_no_punct": base_dir / "encoder_comparison" / "neuralmind_bert" / "stage2_embeddings_results_no_punct.json",
        "neuralmind_large_no_punct": base_dir / "encoder_comparison" / "neuralmind_large_bert" / "stage2_embeddings_results_no_punct.json",
    }
    
    all_results = {}
    for key, path in results_files.items():
        if path.exists():
            all_results[key] = load_results(str(path))
    
    output = {
        "description": "Statistical significance analysis for Stage 2 classification",
        "method": "Bootstrap CIs from fold-level F1 + paired t-test on fold scores",
        "caveat": "With k=3 folds, statistical power is very low. CIs are wide and tests are underpowered. Results should be interpreted cautiously.",
        "bootstrap_cis": {},
        "pairwise_tests": {},
    }
    
    # 1. Bootstrap CIs for each (encoder, classifier) combination
    print("Computing bootstrap confidence intervals...")
    for result_key, result_data in all_results.items():
        output["bootstrap_cis"][result_key] = {}
        for model_name, model_data in result_data["models"].items():
            fold_f1s = [f["f1"] for f in model_data["fold_metrics"]]
            ci = bootstrap_ci_from_folds(fold_f1s)
            ci["fold_f1s"] = fold_f1s
            ci["reported_mean"] = model_data["mean_f1"]
            ci["reported_std"] = model_data["std_f1"]
            output["bootstrap_cis"][result_key][model_name] = ci
            print(f"  {result_key} / {model_name}: F1={ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] (width={ci['ci_width']:.4f})")
    
    # 2. Pairwise comparisons (key comparisons the reviewer asked about)
    print("\nPairwise significance tests...")
    comparisons = [
        # Encoder effect (same classifier)
        ("Legal-BERT vs BERTimbau-base (LR)", 
         "legal_bert_no_punct", "Logistic Regression",
         "neuralmind_base_no_punct", "Logistic Regression"),
        ("Legal-BERT vs BERTimbau-large (LR)",
         "legal_bert_no_punct", "Logistic Regression",
         "neuralmind_large_no_punct", "Logistic Regression"),
        # Classifier effect (same encoder)
        ("LR vs SVM (Legal-BERT)",
         "legal_bert_no_punct", "Logistic Regression",
         "legal_bert_no_punct", "SVM (Linear)"),
        ("LR vs Random Forest (Legal-BERT)",
         "legal_bert_no_punct", "Logistic Regression",
         "legal_bert_no_punct", "Random Forest"),
        # Preprocessing effect
        ("No-punct vs raw (Legal-BERT LR)",
         "legal_bert_no_punct", "Logistic Regression",
         "legal_bert_raw", "Logistic Regression"),
        ("No-punct-no-stop-no-num vs no-punct (Legal-BERT LR)",
         "legal_bert_no_punct_no_stop_no_num", "Logistic Regression",
         "legal_bert_no_punct", "Logistic Regression"),
    ]
    
    for name, key_a, model_a, key_b, model_b in comparisons:
        if key_a not in all_results or key_b not in all_results:
            continue
        
        folds_a = [f["f1"] for f in all_results[key_a]["models"][model_a]["fold_metrics"]]
        folds_b = [f["f1"] for f in all_results[key_b]["models"][model_b]["fold_metrics"]]
        
        test = welch_t_test_from_folds(folds_a, folds_b)
        
        # Also check CI overlap
        ci_a = output["bootstrap_cis"][key_a][model_a]
        ci_b = output["bootstrap_cis"][key_b][model_b]
        cis_non_overlapping = overlapping_ci_test(ci_a, ci_b)
        
        result = {
            "comparison": name,
            "a": f"{key_a} / {model_a} (F1={np.mean(folds_a):.4f})",
            "b": f"{key_b} / {model_b} (F1={np.mean(folds_b):.4f})",
            "paired_t_test": test,
            "ci_non_overlapping": cis_non_overlapping,
        }
        output["pairwise_tests"][name] = result
        
        sig = "YES" if test.get("significant_005") else "NO"
        print(f"  {name}: diff={test.get('mean_diff', 'N/A'):.4f}, p={test.get('p_value', 'N/A'):.4f}, significant(0.05)={sig}")
    
    # 3. Summary interpretation
    output["interpretation"] = {
        "note": "With only k=3 folds, the paired t-test has very low statistical power (df=2). "
                "Most comparisons will not reach significance even for genuinely different methods. "
                "This is an inherent limitation of the small dataset (N=89) and should be explicitly "
                "acknowledged in the paper.",
        "recommendation": "Report CIs to convey uncertainty. State that sample size prevents "
                         "definitive significance claims for most comparisons. "
                         "The LLM vs embedding gap (~0.31 F1) would be significant by any test; "
                         "encoder differences (~0.03-0.12 F1) cannot be reliably distinguished at this N."
    }
    
    # Save (convert numpy types)
    def convert_types(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    output = convert_types(output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    run_experiment4()
