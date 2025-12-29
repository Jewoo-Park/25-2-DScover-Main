import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

import xgboost as xgb
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================
# Feature importance
# ======================================================
def extract_feature_importance(model, X, y, model_name):
    print(f"      [importance] extracting for {model_name}")

    if model_name in ["logreg_l2", "logreg_l1"]:
        return np.abs(model.named_steps["clf"].coef_[0])

    if model_name == "random_forest":
        return model.feature_importances_

    if model_name == "xgboost":
        return model.feature_importances_

    if model_name == "svm_linear":
        return None

    return None


# ======================================================
# Models
# ======================================================
def get_models():
    return {
        "logreg_l2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=500,
                n_jobs=-1
            ))
        ]),

        "logreg_l1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=1000,
                n_jobs=-1
            ))
        ]),

        "svm_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=1.0, max_iter=5000))
        ]),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),

        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }


# ======================================================
# PCA (layer-wise)
# ======================================================
from sklearn.decomposition import PCA

def apply_pca(X, pca_mode, pca_param):
    if pca_mode is None:
        return X

    max_comp = min(X.shape[0], X.shape[1])

    if pca_mode == "fixed":
        n_comp = min(pca_param, max_comp)

        if n_comp < pca_param:
            print(f"    [PCA] n_components reduced: {pca_param} -> {n_comp}")

        print(f"    [PCA] mode=fixed, param={n_comp}")
        pca = PCA(n_components=n_comp, random_state=42)

    elif pca_mode == "variance":
        print(f"    [PCA] mode=ratio, param={pca_param}")
        pca = PCA(n_components=pca_param, random_state=42)

    else:
        raise ValueError(f"Unknown pca_mode: {pca_mode}")

    X_pca = pca.fit_transform(X)
    print(f"    [PCA] shape {X.shape} -> {X_pca.shape}")

    return X_pca



# ======================================================
# Main experiment
# ======================================================
def run_layerwise_experiment(
    df,
    label_col="label",
    layer_col="layer_position",
    pca_mode=None,
    pca_param=None,
    n_splits=3
):
    feature_cols = [c for c in df.columns if c.startswith("dim_")]
    models = get_models()

    results = []
    feature_importances = []

    layers = sorted(df[layer_col].unique())

    print("=" * 90)
    print("[START] Layerwise experiment")
    print(f"  PCA mode  : {pca_mode}")
    print(f"  PCA param : {pca_param}")
    print(f"  CV folds  : {n_splits}")
    print("=" * 90)

    for layer in layers:
        print("\n" + "-" * 90)
        print(f"[Layer {layer}] START")
        layer_start = time.time()

        layer_df = df[df[layer_col] == layer]
        X_raw = layer_df[feature_cols].values
        y = layer_df[label_col].values

        X = apply_pca(X_raw, pca_mode, pca_param)

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        for model_name, model in models.items():
            print(f"\n  [Model: {model_name}]")
            model_start = time.time()

            accs, aucs = [], []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
                print(f"    Fold {fold}/{n_splits}")

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                else:
                    y_score = model.decision_function(X_test)

                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_score)

                accs.append(acc)
                aucs.append(auc)

                print(f"      acc={acc:.4f} | auroc={auc:.4f}")

            mean_acc = np.mean(accs)
            mean_auc = np.mean(aucs)

            model_elapsed = time.time() - model_start
            print(
                f"  → CV result | "
                f"acc={mean_acc:.4f}, "
                f"auroc={mean_auc:.4f}, "
                f"time={model_elapsed/60:.2f} min"
            )

            results.append({
                "layer": layer,
                "model": model_name,
                "pca_mode": pca_mode or "none",
                "pca_param": pca_param,
                "accuracy": mean_acc,
                "auroc": mean_auc
            })

            # # ===== Final importance =====
            # print(f"    [Final fit] for feature importance")
            # model.fit(X, y)
            # imp = extract_feature_importance(model, X, y, model_name)
            #
            # if imp is not None:
            #     feature_importances.append({
            #         "layer": layer,
            #         "model": model_name,
            #         "pca_mode": pca_mode or "none",
            #         "importance": imp
            #     })

        layer_elapsed = time.time() - layer_start
        print(f"[Layer {layer}] DONE in {layer_elapsed/60:.2f} min")

    print("\n" + "=" * 90)
    print("[FINISHED] Experiment completed")
    print("=" * 90)

    return pd.DataFrame(results), feature_importances


# ======================================================
# Entry point
# ======================================================
if __name__ == "__main__":
    print("[INFO] Loading data...")
    df = pd.read_csv("llama3_3b_halueval_representations.csv")
    print(f"[INFO] Data shape: {df.shape}")

    # ---------- RAW ----------
    res_raw, imp_raw = run_layerwise_experiment(df)

    # ---------- PCA 10 ----------
    res_pca10, imp_pca10 = run_layerwise_experiment(
        df,
        pca_mode="fixed",
        pca_param=10
    )

    # ---------- PCA 95% ----------
    res_pca95, imp_pca95 = run_layerwise_experiment(
        df,
        pca_mode="variance",
        pca_param=0.95
    )

    final_results = pd.concat(
        [res_raw, res_pca10, res_pca95],
        ignore_index=True
    )

    final_results.to_csv(
        "layerwise_llama.csv",
        index=False
    )

    print("[INFO] Results saved")
    print("[DONE] 🚀")
