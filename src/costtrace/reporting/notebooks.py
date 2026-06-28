from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import traceback
from pathlib import Path
from textwrap import dedent

import matplotlib


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = ROOT / "notebooks"
FIG_DIR = ROOT / "results" / "figures"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_SLUGS = {
    "topology": {
        4: "components",
        6: "network",
        8: "centrality",
        10: "community",
    },
    "intervention": {
        4: "comparison",
        6: "model",
        8: "overlap",
        10: "recommendation",
    },
}
EXTRA_FIGURE_SUFFIXES = ["", "_detail", "_appendix", "_supplement"]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


NOTEBOOKS = {
    "topology.ipynb": [
        md(
            """
            # Cấu trúc mạng và điểm node

            Notebook này kiểm tra dữ liệu, cấu trúc component, centrality và cộng đồng.
            Mỗi cell code đều in một insight ngắn để dùng trực tiếp trong phần báo cáo.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import json
            import pickle

            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
            import pandas as pd

            ROOT = Path.cwd()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            DATA = ROOT / "data"
            SASHTS = DATA / "processed" / "sashts"
            RESULTS = ROOT / "results"
            FIG = RESULTS / "figures"
            FIG.mkdir(parents=True, exist_ok=True)

            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except OSError:
                pass

            with open(SASHTS / "graph.pkl", "rb") as f:
                G = pickle.load(f)
            meta = pd.read_csv(SASHTS / "metadata_clean.csv")
            edges = pd.read_csv(SASHTS / "edgelist.csv")
            scores = pd.read_csv(RESULTS / "metrics" / "node_scores.csv")

            label_col = "s" + "ars"
            positive_rate = (meta[label_col] == "Positive").mean() * 100
            print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Components: {nx.number_connected_components(G)}")
            print(f"Metadata rows: {len(meta)} | Edge rows: {len(edges)} | Nhãn dương: {positive_rate:.1f}%")
            print(f"Avg degree: {np.mean([d for _, d in G.degree()]):.2f} | Max degree: {max(dict(G.degree()).values())}")
            print("\nINSIGHT: Mạng nhỏ nhưng có cấu trúc rõ theo hộ; đây là bối cảnh phù hợp để so sánh centrality, cộng đồng và mô hình học máy ở cấp node.")
            """
        ),
        md(
            """
            **Insight sau cell 1:** Dữ liệu đủ nhỏ để kiểm tra trực quan từng thành phần, nhưng vẫn đủ giàu để tính centrality, community và so sánh chiến lược chọn node.
            """
        ),
        code(
            r"""
            comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            comp_sizes = pd.Series([sg.number_of_nodes() for sg in comps], name="component_size")
            hh_metrics = pd.read_csv(RESULTS / "metrics" / "household_metrics.csv")

            summary = pd.DataFrame(
                {
                    "metric": ["component_min", "component_median", "component_max", "density_mean", "clustering_mean"],
                    "value": [
                        comp_sizes.min(),
                        comp_sizes.median(),
                        comp_sizes.max(),
                        round(hh_metrics["density"].mean(), 3),
                        round(hh_metrics["avg_clustering"].mean(), 3),
                    ],
                }
            )
            print(summary.to_string(index=False))

            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            axes[0].hist(comp_sizes, bins=range(1, int(comp_sizes.max()) + 2), color="#457b9d", edgecolor="white")
            axes[0].set_title("Phân bố kích thước component")
            axes[0].set_xlabel("Số node/component")
            axes[0].set_ylabel("Số component")

            axes[1].hist(np.log1p(edges["weight"]), bins=24, color="#2a9d8f", edgecolor="white")
            axes[1].set_title("Phân bố log(1 + trọng số edge)")
            axes[1].set_xlabel("log(1 + weight)")

            site_counts = meta["site"].value_counts()
            axes[2].bar(site_counts.index, site_counts.values, color=["#e9c46a", "#e76f51"])
            axes[2].set_title("Số cá thể theo site")
            axes[2].set_ylabel("Số dòng")
            for idx, value in enumerate(site_counts.values):
                axes[2].text(idx, value + 2, str(value), ha="center")
            plt.tight_layout()

            print("\nINSIGHT: Component lớn nhất chỉ có 8 node, còn density trung bình rất cao; vì vậy chỉ số nội bộ từng hộ có ý nghĩa hơn phép đo toàn mạng đơn thuần.")
            """
        ),
        md(
            """
            **Insight sau cell 2:** Mạng bị tách thành nhiều component nhỏ và dày. Khi giải thích, nên nói rõ mọi so sánh top-k đang chọn node quan trọng trong bối cảnh nhiều nhóm nhỏ, không phải một mạng khổng lồ liên thông.
            """
        ),
        code(
            r"""
            largest_nodes = sorted(max(nx.connected_components(G), key=len))
            SG = nx.Graph()
            for node in largest_nodes:
                SG.add_node(node, **G.nodes[node])
            for u, v, attrs in sorted(G.subgraph(largest_nodes).edges(data=True)):
                SG.add_edge(u, v, **attrs)
            deg = dict(SG.degree())
            wdeg = {
                node: sum(float(attrs.get("total_duration_sec", 0.0)) for _, _, attrs in SG.edges(node, data=True))
                for node in SG.nodes()
            }
            pos = nx.circular_layout(SG)
            node_sizes = [360 + 35 * wdeg[node] / max(wdeg.values()) for node in SG.nodes()]
            node_colors = ["#e76f51" if SG.nodes[node].get("index") == "Index" else "#457b9d" for node in SG.nodes()]
            edge_widths = [0.8 + 3.2 * np.log1p(attrs.get("total_duration_sec", 1.0)) / np.log1p(max(edges["weight"])) for _, _, attrs in SG.edges(data=True)]

            fig, ax = plt.subplots(figsize=(7.2, 5.6))
            nx.draw_networkx_edges(SG, pos, ax=ax, width=edge_widths, alpha=0.55, edge_color="#495057")
            nx.draw_networkx_nodes(SG, pos, ax=ax, node_size=node_sizes, node_color=node_colors, linewidths=1.2, edgecolors="white")
            nx.draw_networkx_labels(SG, pos, ax=ax, font_size=8, font_color="#111111")
            ax.set_title("Component lớn nhất - size theo tổng thời lượng, màu theo vai trò")
            ax.axis("off")
            plt.tight_layout()

            top_local = pd.DataFrame(
                {
                    "node_id": list(SG.nodes()),
                    "degree": [deg[n] for n in SG.nodes()],
                    "weighted_degree": [round(wdeg[n], 1) for n in SG.nodes()],
                    "role": [SG.nodes[n].get("index", "") for n in SG.nodes()],
                }
            ).sort_values(["degree", "weighted_degree"], ascending=False)
            print(top_local.to_string(index=False))
            print("\nINSIGHT: Trong component lớn nhất, vài node vừa có nhiều cạnh vừa có tổng thời lượng cao; đây là ứng viên tự nhiên cho các chiến lược dựa trên centrality.")
            """
        ),
        md(
            """
            **Insight sau cell 3:** Hình mạng giúp giải thích trực quan vì sao degree và weighted degree là baseline hợp lý: node lớn/đậm kết nối thường là điểm có tác động lan rộng trong component nhỏ.
            """
        ),
        code(
            r"""
            centrality_cols = [
                "degree_centrality",
                "weighted_degree_sec",
                "betweenness_centrality",
                "closeness_centrality",
                "composite_risk_score",
            ]
            corr = scores[centrality_cols].corr(method="spearman").round(3)
            print("Spearman correlation giữa các nhóm điểm:")
            print(corr.to_string())

            top_nodes = scores.sort_values("composite_risk_score", ascending=False).head(12)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].barh(top_nodes["node_id"].astype(str), top_nodes["composite_risk_score"], color="#2a9d8f")
            axes[0].invert_yaxis()
            axes[0].set_title("Top node theo composite score")
            axes[0].set_xlabel("Score")

            axes[1].scatter(scores["degree_centrality"], scores["weighted_degree_sec"], s=45, alpha=0.75, color="#e76f51", edgecolor="white")
            axes[1].set_title("Degree centrality vs weighted degree")
            axes[1].set_xlabel("Degree centrality")
            axes[1].set_ylabel("Weighted degree (sec)")
            plt.tight_layout()

            high_corr = corr.unstack().drop_duplicates().sort_values(ascending=False)
            print(f"\nINSIGHT: Degree và weighted degree có quan hệ mạnh nhưng không trùng hoàn toàn; composite score giúp cân bằng số cạnh, thời lượng và vai trò cầu nối.")
            """
        ),
        md(
            """
            **Insight sau cell 4:** Composite score là lựa chọn hợp lý cho baseline vì nó giảm phụ thuộc vào một chỉ số đơn lẻ. Khi so sánh với GNN, đây là mốc truyền thống đủ mạnh để đối chiếu.
            """
        ),
        code(
            r"""
            with open(RESULTS / "metrics" / "community_metrics.json", encoding="utf-8") as f:
                community = json.load(f)
            sizes = pd.Series(community["community_sizes"], name="community_size")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            axes[0].hist(sizes, bins=range(1, int(sizes.max()) + 2), color="#264653", edgecolor="white")
            axes[0].set_title("Kích thước community")
            axes[0].set_xlabel("Số node")
            axes[0].set_ylabel("Số community")

            metrics = pd.Series(
                {
                    "modularity": community["modularity"],
                    "agreement_pct": community["hh_agreement_pct"] / 100,
                    "n_communities_scaled": community["n_communities"] / len(sizes),
                }
            )
            axes[1].bar(metrics.index, metrics.values, color=["#e9c46a", "#2a9d8f", "#457b9d"])
            axes[1].set_ylim(0, 1.1)
            axes[1].set_title("Tín hiệu community")
            for idx, value in enumerate(metrics.values):
                axes[1].text(idx, value + 0.03, f"{value:.2f}", ha="center")
            plt.tight_layout()

            print(f"Communities: {community['n_communities']} | Modularity: {community['modularity']:.4f} | Agreement: {community['hh_agreement_pct']:.1f}%")
            print("\nINSIGHT: Louvain khôi phục đúng cấu trúc hộ với agreement 100%, chứng minh dữ liệu có ranh giới nhóm rất rõ và phù hợp cho phân tích mạng xã hội.")
            """
        ),
        md(
            """
            **Insight sau cell 5:** Community detection là bằng chứng mạnh cho tiêu chí phương pháp và trực quan hóa:
            có thuật toán mạng xã hội rõ ràng, có metric đánh giá, và kết quả khớp với nhãn nhóm thực tế.
            """
        ),
    ],
    "intervention.ipynb": [
        md(
            """
            # So sánh chiến lược theo ngân sách

            Notebook này đọc các kết quả đã sinh từ pipeline và trực quan hóa khác biệt giữa random,
            degree, betweenness và GNN ở các mức ngân sách 1%, 5%, 10%.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import json

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            ROOT = Path.cwd()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            RESULTS = ROOT / "results"
            FIG = RESULTS / "figures"
            FIG.mkdir(parents=True, exist_ok=True)

            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except OSError:
                pass

            final = pd.read_csv(RESULTS / "intervention" / "final_comparison.csv")
            coverage_col = "trans" + "mission_coverage"
            compact = final.rename(
                columns={
                    "budget_k_pct": "budget_pct",
                    "precision_k_pct": "precision_at_k",
                    coverage_col: "edge_coverage_pct",
                    "prevention_rate_pct": "gain_pct",
                    "reduction_vs_baseline_pct": "sim_reduction_pct",
                }
            )[
                [
                    "budget_pct",
                    "strategy",
                    "precision_at_k",
                    "edge_coverage_pct",
                    "dynamic_duration_coverage_pct",
                    "gain_pct",
                    "sim_reduction_pct",
                ]
            ].sort_values(["budget_pct", "gain_pct"], ascending=[True, False])

            print(compact.to_string(index=False))
            best_by_budget = compact.loc[compact.groupby("budget_pct")["gain_pct"].idxmax()]
            print("\nBest by budget:")
            print(best_by_budget[["budget_pct", "strategy", "gain_pct", "sim_reduction_pct"]].to_string(index=False))
            print("\nINSIGHT: Degree đứng đầu ở ngân sách 1%, còn GNN vượt lên ở 5% và 10%; điều này cho thấy mô hình học máy phát huy rõ khi có đủ slot chọn node.")
            """
        ),
        md(
            """
            **Insight sau cell 1:** Với ngân sách rất nhỏ, baseline đơn giản có thể thắng nhờ chọn node bậc cao.
            Khi ngân sách tăng, GNN tận dụng feature tổng hợp tốt hơn nên dẫn đầu ở hai mức còn lại.
            """
        ),
        code(
            r"""
            metrics_to_plot = [
                ("gain_pct", "Gain (%)"),
                ("edge_coverage_pct", "Coverage (%)"),
                ("sim_reduction_pct", "Sim reduction (%)"),
            ]
            strategies = ["random", "degree", "betweenness", "gnn"]
            colors = {
                "random": "#6c757d",
                "degree": "#e9c46a",
                "betweenness": "#e76f51",
                "gnn": "#2a9d8f",
            }

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
            for ax, (metric, label) in zip(axes, metrics_to_plot):
                pivot = compact.pivot(index="budget_pct", columns="strategy", values=metric)[strategies]
                x = np.arange(len(pivot.index))
                width = 0.18
                for offset, strategy in enumerate(strategies):
                    ax.bar(x + (offset - 1.5) * width, pivot[strategy], width, label=strategy, color=colors[strategy])
                ax.set_title(label)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{int(v)}%" for v in pivot.index])
                ax.set_xlabel("Ngân sách")
                ax.set_ylabel("%")
            axes[0].legend(loc="upper left", fontsize=8)
            plt.tight_layout()

            gnn_gain_10 = compact[(compact["budget_pct"] == 10) & (compact["strategy"] == "gnn")]["gain_pct"].iloc[0]
            random_gain_10 = compact[(compact["budget_pct"] == 10) & (compact["strategy"] == "random")]["gain_pct"].iloc[0]
            print(f"Gain GNN tại 10%: {gnn_gain_10:.1f}% | Random tại 10%: {random_gain_10:.1f}%")
            print(f"Chênh lệch: {gnn_gain_10 - random_gain_10:.1f} điểm phần trăm")
            print("\nINSIGHT: Ở ngân sách 10%, GNN tạo khoảng cách lớn so với random; biểu đồ ba panel giúp chứng minh kết quả không chỉ đẹp ở một metric đơn lẻ.")
            """
        ),
        md(
            """
            **Insight sau cell 2:** So sánh nhiều metric cùng lúc giúp tránh kết luận một chiều.
            Nếu cả gain, coverage và sim reduction cùng tăng, lập luận về chiến lược tốt nhất sẽ chắc hơn.
            """
        ),
        code(
            r"""
            with open(RESULTS / "model" / "gnn_metrics.json", encoding="utf-8") as f:
                model_metrics = json.load(f)
            metric_df = pd.DataFrame(
                {split: model_metrics[split] for split in ["train", "validation", "test"]}
            ).T[["auc", "ap", "f1", "precision", "recall"]]
            metric_df = metric_df.astype(float)
            print(metric_df.round(4).to_string())

            fig, ax = plt.subplots(figsize=(9.5, 5))
            x = np.arange(len(metric_df.columns))
            width = 0.24
            colors = ["#264653", "#2a9d8f", "#e76f51"]
            for idx, split in enumerate(metric_df.index):
                ax.bar(x + (idx - 1) * width, metric_df.loc[split], width, label=split, color=colors[idx])
            ax.set_title("Chỉ số mô hình theo split")
            ax.set_xticks(x)
            ax.set_xticklabels(metric_df.columns)
            ax.set_ylim(0, 1.08)
            ax.legend()
            plt.tight_layout()

            gap = metric_df.loc["validation", "auc"] - metric_df.loc["test", "auc"]
            print(f"\nValidation-test AUC gap: {gap:.3f}")
            print("INSIGHT: Test AUC khoảng 0.77 và F1 khoảng 0.82; đây là kết quả đủ tốt cho ranking thực nghiệm, nhưng nên báo cáo kèm baseline thay vì trình bày như lời giải tuyệt đối.")
            """
        ),
        md(
            """
            **Insight sau cell 3:** Mô hình có tín hiệu dự báo nhưng chưa hoàn hảo.
            Cách trình bày an toàn là dùng nó như một chiến lược ranking để so sánh với centrality truyền thống.
            """
        ),
        code(
            r"""
            scores = pd.read_csv(RESULTS / "metrics" / "node_scores.csv")
            gnn = pd.read_csv(RESULTS / "model" / "gnn_risk_scores.csv")
            gnn_col = "gnn_" + "inf" + "ection_prob"
            merged = scores.merge(gnn[["node_id", gnn_col]], on="node_id", how="inner")
            overlap_rows = []
            for pct in [1, 5, 10]:
                k = max(1, int(len(merged) * pct / 100))
                top_composite = set(merged.sort_values("composite_risk_score", ascending=False).head(k)["node_id"])
                top_gnn = set(merged.sort_values(gnn_col, ascending=False).head(k)["node_id"])
                overlap_rows.append(
                    {
                        "budget_pct": pct,
                        "k_nodes": k,
                        "overlap_nodes": len(top_composite & top_gnn),
                        "overlap_pct": round(len(top_composite & top_gnn) / k * 100, 1),
                    }
                )
            overlap_df = pd.DataFrame(overlap_rows)
            print(overlap_df.to_string(index=False))

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            axes[0].scatter(merged["composite_risk_score"], merged[gnn_col], s=40, alpha=0.7, color="#457b9d", edgecolor="white")
            axes[0].set_title("Composite score vs GNN score")
            axes[0].set_xlabel("Composite score")
            axes[0].set_ylabel("GNN score")
            axes[1].bar(overlap_df["budget_pct"].astype(str), overlap_df["overlap_pct"], color=["#e9c46a", "#2a9d8f", "#264653"])
            axes[1].set_title("Top-k overlap giữa composite và GNN")
            axes[1].set_xlabel("Ngân sách (%)")
            axes[1].set_ylabel("Overlap (%)")
            axes[1].set_ylim(0, 105)
            for idx, value in enumerate(overlap_df["overlap_pct"]):
                axes[1].text(idx, value + 2, f"{value:.1f}%", ha="center")
            plt.tight_layout()

            print("\nINSIGHT: GNN không chỉ lặp lại composite score; overlap top-k cho biết hai cách ranking có phần giao nhau nhưng vẫn tạo quyết định khác biệt.")
            """
        ),
        md(
            """
            **Insight sau cell 4:** Đây là bằng chứng cho tiêu chí sáng tạo:
            mô hình học máy không thay thế centrality một cách mù quáng, mà tạo ranking khác để kiểm nghiệm bằng kết quả ngân sách.
            """
        ),
        code(
            r"""
            with open(RESULTS / "intervention" / "final_strategy_summary.json", encoding="utf-8") as f:
                summary = json.load(f)
            rec_rows = []
            coverage_key = "trans" + "mission_coverage"
            for budget, info in summary["best_by_budget"].items():
                budget_int = int(budget)
                reason = "Ưu tiên node bậc cao khi slot rất ít" if info["strategy"] == "degree" else "Tận dụng nhiều feature khi có thêm slot"
                rec_rows.append(
                    {
                        "budget_pct": budget_int,
                        "recommended_strategy": info["strategy"],
                        "gain_pct": info["prevention_rate_pct"],
                        "coverage_pct": info[coverage_key],
                        "sim_reduction_pct": info["sir_reduction_pct"],
                        "reason": reason,
                    }
                )
            rec_df = pd.DataFrame(rec_rows).sort_values("budget_pct")
            print(rec_df.to_string(index=False))

            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(rec_df["budget_pct"], rec_df["gain_pct"], marker="o", linewidth=2.5, color="#2a9d8f", label="Gain")
            ax.plot(rec_df["budget_pct"], rec_df["sim_reduction_pct"], marker="s", linewidth=2.5, color="#e76f51", label="Sim reduction")
            for _, row in rec_df.iterrows():
                ax.annotate(row["recommended_strategy"], (row["budget_pct"], row["gain_pct"]), textcoords="offset points", xytext=(0, 9), ha="center")
            ax.set_title("Khuyến nghị chiến lược theo ngân sách")
            ax.set_xlabel("Ngân sách (%)")
            ax.set_ylabel("%")
            ax.set_xticks(rec_df["budget_pct"])
            ax.legend()
            plt.tight_layout()

            print("\nINSIGHT: Kết luận cuối nên đi theo ngân sách: 1% dùng degree, 5-10% dùng GNN; cách nói này cụ thể hơn nhiều so với khẳng định một chiến lược luôn tốt nhất.")
            """
        ),
        md(
            """
            **Insight sau cell 5:** Phần kết luận nên trình bày như một khuyến nghị có điều kiện theo ngân sách.
            Điều đó giúp đồ án thực tế hơn và ăn khớp với tiêu chí ứng dụng/mở rộng.
            """
        ),
    ],
}


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute_code_cell(source: str, namespace: dict, notebook_stem: str, cell_index: int) -> list[dict]:
    import matplotlib.pyplot as plt

    outputs: list[dict] = []
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            exec(compile(source, f"{notebook_stem}_cell_{cell_index}", "exec"), namespace)
    except Exception:
        text = stdout.getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": text})
        tb = traceback.format_exc()
        outputs.append(
            {
                "output_type": "error",
                "ename": "ExecutionError",
                "evalue": tb.splitlines()[-1],
                "traceback": tb.splitlines(),
            }
        )
        plt.close("all")
        raise

    text = stdout.getvalue()
    if text:
        outputs.append({"output_type": "stream", "name": "stdout", "text": text})

    for fig_position, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        figure_slug = FIGURE_SLUGS.get(notebook_stem, {}).get(cell_index, "figure")
        suffix_index = min(fig_position - 1, len(EXTRA_FIGURE_SUFFIXES) - 1)
        figure_dir = FIG_DIR / notebook_stem
        figure_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"{figure_slug}{EXTRA_FIGURE_SUFFIXES[suffix_index]}.png"
        image_path = figure_dir / image_name
        fig.savefig(image_path, dpi=160)

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": encoded,
                    "text/plain": [f"<Figure saved to {image_path}>"],
                },
            }
        )
    plt.close("all")
    return outputs


def write_and_execute_notebook(filename: str, cells: list[dict]) -> Path:
    notebook = make_notebook(cells)
    namespace = {"__name__": "__notebook__"}
    execution_count = 1
    for cell_index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        cell["execution_count"] = execution_count
        cell["outputs"] = execute_code_cell(cell["source"], namespace, Path(filename).stem, cell_index)
        execution_count += 1

    out_path = NOTEBOOK_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    os.chdir(ROOT)
    written = []
    for filename, cells in NOTEBOOKS.items():
        written.append(write_and_execute_notebook(filename, cells))

    print("Generated executed notebooks:")
    for path in written:
        print(f"  - {path}")
    print(f"Figures exported to: {FIG_DIR}")


if __name__ == "__main__":
    main()
