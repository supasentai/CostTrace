import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "audit"
DOCS_DIR = ROOT / "docs" / "paper_upgrade"


def legacy_node_level_split(labels: np.ndarray) -> pd.Series:
    rng = np.random.default_rng(SEED)
    split = pd.Series([""] * len(labels), dtype="object")

    for label in [0, 1]:
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_train = int(0.70 * len(idx))
        n_val = int(0.15 * len(idx))
        split.iloc[idx[:n_train]] = "train"
        split.iloc[idx[n_train : n_train + n_val]] = "validation"
        split.iloc[idx[n_train + n_val :]] = "test"

    return split


def summarize_proxy_feature(df: pd.DataFrame, feature: str) -> list[dict[str, float]]:
    rows = []
    for value, group in df.groupby(feature, dropna=False):
        rows.append(
            {
                "feature": feature,
                "value": str(value),
                "n": int(len(group)),
                "sars_positive": int(group["sars_label"].sum()),
                "positive_rate": round(float(group["sars_label"].mean()), 4),
            }
        )
    return rows


def split_household_overlap(df: pd.DataFrame, split_col: str) -> dict[str, int]:
    split_sets = df.groupby("hhid")[split_col].agg(lambda values: set(values))
    return {
        "households_total": int(split_sets.size),
        "households_with_train_test_overlap": int(
            sum({"train", "test"}.issubset(values) for values in split_sets)
        ),
        "households_with_any_cross_split_overlap": int(sum(len(values) > 1 for values in split_sets)),
    }


def current_split_from_metadata(df: pd.DataFrame) -> pd.Series | None:
    metadata_path = ROOT / "models" / "gnn_metadata.json"
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_households = metadata.get("split_households", {})
    if not split_households:
        return None

    household_to_split = {}
    for split_name, households in split_households.items():
        normalized = "validation" if split_name == "validation" else split_name
        for hhid in households:
            household_to_split[hhid] = normalized

    return df["hhid"].map(household_to_split).fillna("unassigned")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    raw_contacts = pd.read_csv(ROOT / "data" / "raw" / "sashts_contact_network.csv")
    raw_meta = pd.read_csv(ROOT / "data" / "raw" / "sashts_metadata.csv")
    edges = pd.read_csv(ROOT / "data" / "processed" / "edgelist.csv")
    node_scores = pd.read_csv(ROOT / "results" / "metrics" / "node_scores.csv")
    node_scores = node_scores.sort_values("node_id").reset_index(drop=True)

    legacy_split = legacy_node_level_split(node_scores["sars_label"].to_numpy())
    node_scores["legacy_node_split"] = legacy_split

    current_split = current_split_from_metadata(node_scores)
    if current_split is not None:
        node_scores["current_household_split"] = current_split

    proxy_features = ["is_index", "sus_enc", "sleep_room_enc", "cared_by_enc"]
    proxy_summary = []
    for feature in proxy_features:
        proxy_summary.extend(summarize_proxy_feature(node_scores, feature))

    dictionary = pd.read_excel(ROOT / "data" / "raw" / "Data dictionary.xlsx")
    dictionary = dictionary.rename(columns={"Vairable": "Variable"})
    selected_variables = [
        "indid",
        "site",
        "agegrp9",
        "sex",
        "hhid",
        "sleep_room_ix",
        "cared_by_ix",
        "sus",
        "index",
        "sars",
        "duration_sec",
        "no_ts",
        "pair_sars",
    ]
    dictionary_out = dictionary[dictionary["Variable"].isin(selected_variables)].copy()
    dictionary_out.to_csv(AUDIT_DIR / "phase03_data_dictionary_mapping.csv", index=False)

    report = {
        "dataset": {
            "raw_proximity_events": int(len(raw_contacts)),
            "metadata_rows": int(len(raw_meta)),
            "individuals": int(raw_meta["indid"].nunique()),
            "households": int(raw_meta["hhid"].nunique()),
            "processed_edges": int(len(edges)),
            "self_loops_in_raw": int((raw_contacts["indid1"] == raw_contacts["indid2"]).sum()),
            "real_timestamp_rows": int((raw_contacts["no_ts"] == 0).sum()),
            "missing_real_timestamp_rows": int((raw_contacts["no_ts"] != 0).sum()),
            "sites": {str(k): int(v) for k, v in raw_meta["site"].value_counts().items()},
        },
        "legacy_node_split_risk": split_household_overlap(node_scores, "legacy_node_split"),
        "proxy_feature_summary": proxy_summary,
        "mitigation": {
            "removed_from_graphsage_features": ["is_index", "sus_enc"],
            "required_split_protocol": "household-level split with no household overlap across train/validation/test",
            "required_scaling_protocol": "fit continuous-feature scaling parameters on train households only",
        },
        "artifacts": {
            "data_dictionary_mapping": "results/audit/phase03_data_dictionary_mapping.csv",
        },
    }

    if current_split is not None:
        report["current_household_split_check"] = split_household_overlap(
            node_scores, "current_household_split"
        )

    (AUDIT_DIR / "phase03_leakage_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    proxy_df = pd.DataFrame(proxy_summary)
    proxy_df.to_csv(AUDIT_DIR / "phase03_proxy_feature_summary.csv", index=False)

    markdown_lines = [
        "# Phase 03 Leakage Audit Table",
        "",
        "Generated by `scripts/18_phase03_leakage_audit.py`.",
        "",
        "| Risk | Evidence | Mitigation | Residual risk |",
        "|---|---|---|---|",
        (
            "| Label-proxy feature leakage | `is_index` and `sus_enc` show deterministic "
            "alignment with SARS positivity in the current artifacts; see "
            "`results/audit/phase03_proxy_feature_summary.csv`. | Removed from "
            "GraphSAGE input features. | Other epidemiological variables may still be "
            "post-outcome unless the observation time is confirmed. |"
        ),
        (
            "| Household overlap across node-level split | Legacy node split has "
            f"{report['legacy_node_split_risk']['households_with_train_test_overlap']} "
            "households containing both train and test nodes. | Use household-level "
            "split with no household overlap. | Small number of households may increase "
            "variance; multi-fold evaluation remains a Phase 04 task. |"
        ),
        (
            "| Full-data feature scaling | Previous normalized columns were produced before "
            "the model split. | GraphSAGE now scales continuous features from train "
            "households only. | Existing composite risk score remains descriptive and "
            "should not be interpreted as a held-out predictive model. |"
        ),
        (
            "| Edge outcome leakage | Processed edge lists retain `pair_sars` / "
            "`transmission` for counterfactual evaluation. | GraphSAGE adjacency uses "
            "only contact duration weights. | Downstream intervention metrics still use "
            "observed transmission labels by design and must be framed as retrospective. |"
        ),
    ]
    (DOCS_DIR / "phase_03_leakage_audit_table.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )

    print("Wrote results/audit/phase03_leakage_audit.json")
    print("Wrote results/audit/phase03_proxy_feature_summary.csv")
    print("Wrote results/audit/phase03_data_dictionary_mapping.csv")
    print("Wrote docs/paper_upgrade/phase_03_leakage_audit_table.md")


if __name__ == "__main__":
    main()
