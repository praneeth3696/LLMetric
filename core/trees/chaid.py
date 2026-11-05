"""LLMetrics Chi-Square Automatic Interaction Detection (CHAID) Module.

A pure-Python, scipy-based implementation of CHAID decision tree induction for
categorical and binned continuous predictors on binary outcomes.
Uses scipy.stats.chi2_contingency to select optimal multiway splits by chi-square significance.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

class CHAIDNode:
    """Represents a node in a CHAID decision tree."""
    def __init__(
        self,
        node_id: int,
        depth: int = 0,
        parent_rule: Optional[str] = None,
    ):
        self.node_id = node_id
        self.depth = depth
        self.parent_rule = parent_rule
        self.split_feature: Optional[str] = None
        self.chi2_stat: Optional[float] = None
        self.p_value: Optional[float] = None
        self.dof: Optional[int] = None
        self.children: Dict[str, "CHAIDNode"] = {}
        self.is_leaf: bool = True
        self.sample_count: int = 0
        self.class_counts: Dict[int, int] = {0: 0, 1: 0}
        self.predicted_class: int = 0
        self.failure_probability: float = 0.0

class CHAIDClassifier:
    """CHAID Decision Tree Classifier for binary failure prediction."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 30,
        min_samples_leaf: int = 15,
        alpha_merge: float = 0.05,
        num_bins: int = 4,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.alpha_merge = alpha_merge
        self.num_bins = num_bins
        self.root: Optional[CHAIDNode] = None
        self.bin_edges_: Dict[str, List[float]] = {}
        self.feature_names_: List[str] = []
        self._node_counter = 0

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> "CHAIDClassifier":
        """Fits the CHAID tree on the specified DataFrame."""
        self._node_counter = 0
        self.feature_names_ = feature_cols
        self.bin_edges_ = {}

        # Discretize continuous columns
        binned_df = df.copy()
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(binned_df[col]) and binned_df[col].nunique() > self.num_bins:
                try:
                    labels = [f"Bin_{i+1}" for i in range(self.num_bins)]
                    binned_series, edges = pd.qcut(
                        binned_df[col],
                        q=self.num_bins,
                        labels=labels,
                        retbins=True,
                        duplicates="drop",
                    )
                    binned_df[col] = binned_series.astype(str)
                    self.bin_edges_[col] = [float(e) for e in edges]
                except Exception:
                    # Fallback to equal-width cuts if quantile cut has too many duplicate ties
                    labels = [f"Bin_{i+1}" for i in range(self.num_bins)]
                    binned_series, edges = pd.cut(
                        binned_df[col],
                        bins=self.num_bins,
                        labels=labels,
                        retbins=True,
                    )
                    binned_df[col] = binned_series.astype(str)
                    self.bin_edges_[col] = [float(e) for e in edges]
            else:
                binned_df[col] = binned_df[col].astype(str)

        self.root = self._build_tree(binned_df, target_col, feature_cols, depth=0)
        return self

    def _build_tree(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: List[str],
        depth: int,
        parent_rule: Optional[str] = None,
    ) -> CHAIDNode:
        node = CHAIDNode(node_id=self._node_counter, depth=depth, parent_rule=parent_rule)
        self._node_counter += 1

        n_samples = len(df)
        node.sample_count = n_samples

        if n_samples == 0:
            return node

        y = df[target_col].astype(int)
        c0 = int((y == 0).sum())
        c1 = int((y == 1).sum())
        node.class_counts = {0: c0, 1: c1}
        node.predicted_class = 1 if c1 > c0 else 0
        node.failure_probability = float(c1 / n_samples)

        # Stop criteria: pure node, max depth, or small node
        if c0 == 0 or c1 == 0 or depth >= self.max_depth or n_samples < self.min_samples_split:
            node.is_leaf = True
            return node

        # Find best splitting feature using Chi-square test of independence
        best_feat = None
        best_p_val = 1.0
        best_chi2 = 0.0
        best_dof = 1

        for feat in features:
            contingency = pd.crosstab(df[feat], y)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            try:
                chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
                if p_val < best_p_val and p_val < self.alpha_merge:
                    best_p_val = p_val
                    best_chi2 = chi2
                    best_dof = dof
                    best_feat = feat
            except Exception:
                continue

        if best_feat is None:
            node.is_leaf = True
            return node

        # Successful split found
        node.is_leaf = False
        node.split_feature = best_feat
        node.chi2_stat = float(best_chi2)
        node.p_value = float(best_p_val)
        node.dof = int(best_dof)

        remaining_features = [f for f in features if f != best_feat]
        groups = df.groupby(best_feat, observed=False)

        for category, grp in groups:
            if len(grp) < self.min_samples_leaf:
                continue
            child_rule = f"{best_feat} = {category}"
            child_node = self._build_tree(
                grp, target_col, remaining_features, depth=depth + 1, parent_rule=child_rule
            )
            node.children[str(category)] = child_node

        if not node.children:
            node.is_leaf = True

        return node

    def _transform_input_row(self, row: pd.Series) -> Dict[str, str]:
        """Maps continuous values of a single row into discrete category bins."""
        out = {}
        for feat in self.feature_names_:
            val = row[feat]
            if feat in self.bin_edges_:
                edges = self.bin_edges_[feat]
                # Determine bin index
                bin_idx = self.num_bins
                for i in range(len(edges) - 1):
                    if val <= edges[i + 1]:
                        bin_idx = i + 1
                        break
                out[feat] = f"Bin_{bin_idx}"
            else:
                out[feat] = str(val)
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts class failure probabilities for DataFrame rows."""
        probs = []
        for _, row in df.iterrows():
            row_dict = self._transform_input_row(row)
            curr = self.root
            while curr is not None and not curr.is_leaf:
                split_val = row_dict.get(curr.split_feature)
                if split_val in curr.children:
                    curr = curr.children[split_val]
                else:
                    break
            probs.append(curr.failure_probability if curr else 0.0)
        return np.array(probs)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predicts class labels for DataFrame rows."""
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)

    def get_rules(self) -> List[str]:
        """Extracts human-readable decision rules from the CHAID tree."""
        rules: List[str] = []

        def _traverse(node: CHAIDNode, path_rules: List[str]) -> None:
            if node.is_leaf or not node.children:
                condition_str = " AND ".join(path_rules) if path_rules else "All Requests"
                rules.append(
                    f"IF {condition_str} THEN P(Failure) = {node.failure_probability:.1%} "
                    f"(n = {node.sample_count}, Pred = {'Fail' if node.predicted_class == 1 else 'Success'})"
                )
            else:
                for cat, child in node.children.items():
                    edge_desc = f"{node.split_feature} == '{cat}'"
                    _traverse(child, path_rules + [edge_desc])

        if self.root:
            _traverse(self.root, [])
        return rules

def fit_chaid_classifier(
    df: pd.DataFrame,
    max_depth: int = 3,
    min_samples_split: int = 30,
    min_samples_leaf: int = 15,
) -> Dict[str, Any]:
    """Helper function to fit and evaluate CHAID on a dataset."""
    sub_df = df.copy()
    sub_df["is_failure"] = (sub_df["status"] != "ok").astype(int)

    if "is_peak_hour" not in sub_df.columns:
        if "timestamp" in sub_df.columns:
            ts = pd.to_datetime(sub_df["timestamp"], utc=True)
            sub_df["is_peak_hour"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 18)).astype(int)
        else:
            sub_df["is_peak_hour"] = 0

    features = ["latency_ms", "prompt_tokens", "completion_tokens", "model", "is_peak_hour", "cache_hit"]
    chaid = CHAIDClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    chaid.fit(sub_df, target_col="is_failure", feature_cols=features)

    y_true = sub_df["is_failure"].values
    y_prob = chaid.predict_proba(sub_df)
    y_pred = (y_prob >= 0.30).astype(int)

    acc = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "model_name": "CHAID Decision Tree (Chi-Square Automatic Interaction Detection)",
        "hyperparameters": {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "alpha_merge": chaid.alpha_merge,
            "num_bins": chaid.num_bins,
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
        },
        "human_readable_rules": chaid.get_rules(),
        "bin_definitions": chaid.bin_edges_,
    }
