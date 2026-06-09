from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualizations"
GEPHI_DIR = ROOT / "results" / "gephi"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = str(value).strip().lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    GEPHI_DIR.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(VIS_DIR / "gephi_nodes.csv")
    edges = pd.read_csv(VIS_DIR / "gephi_edges.csv")

    # Tao moi anh PNG bang Python tu export Gephi/Louvain vi khong dung Gephi GUI/Chocolatey.
    # Mau node lay truc tiep tu cot Community Color da gan theo Louvain Community Id.
    nodes["Community Id"] = nodes["Community Id"].astype(int)
    nodes["Degree"] = pd.to_numeric(nodes["Degree"], errors="coerce").fillna(1)
    nodes["Community Color"] = nodes["Community Color"].fillna("#999999")

    width, height = 2600, 1800
    scale = 2
    canvas = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(34 * scale, bold=True)
    subtitle_font = load_font(19 * scale)
    label_font = load_font(12 * scale)
    small_font = load_font(15 * scale)

    communities = sorted(nodes["Community Id"].unique())
    n_cols = 11
    n_rows = math.ceil(len(communities) / n_cols)
    margin_x, top_y, bottom_y = 90 * scale, 170 * scale, 120 * scale
    cell_w = (width * scale - 2 * margin_x) / n_cols
    cell_h = (height * scale - top_y - bottom_y) / n_rows

    positions: dict[str, tuple[float, float]] = {}
    community_centers: dict[int, tuple[float, float]] = {}
    max_degree = max(float(nodes["Degree"].max()), 1.0)

    for idx, community_id in enumerate(communities):
        row, col = divmod(idx, n_cols)
        cx = margin_x + cell_w * (col + 0.5)
        cy = top_y + cell_h * (row + 0.5)
        community_centers[community_id] = (cx, cy)

        group = nodes[nodes["Community Id"] == community_id].sort_values("Id")
        n = len(group)
        radius = min(cell_w, cell_h) * (0.16 + min(n, 8) * 0.018)

        for local_idx, (_, node) in enumerate(group.iterrows()):
            if n == 1:
                x, y = cx, cy
            else:
                angle = 2 * math.pi * local_idx / n - math.pi / 2
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
            positions[str(node["Id"])] = (x, y)

    edge_weight = pd.to_numeric(edges["Weight"], errors="coerce").fillna(0)
    weight_q95 = max(float(edge_weight.quantile(0.95)), 1.0)

    for _, edge in edges.iterrows():
        source, target = str(edge["Source"]), str(edge["Target"])
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        weight = float(edge["Weight"])
        line_width = int(max(1, min(7, 1 + 6 * math.sqrt(weight / weight_q95))) * scale)
        draw.line((x1, y1, x2, y2), fill=(164, 171, 179), width=line_width)

    for community_id in communities:
        group = nodes[nodes["Community Id"] == community_id].sort_values("Id")
        cx, cy = community_centers[community_id]
        color = hex_to_rgb(group.iloc[0]["Community Color"])
        draw.ellipse(
            (
                cx - 5 * scale,
                cy - 5 * scale,
                cx + 5 * scale,
                cy + 5 * scale,
            ),
            fill=(235, 235, 235),
            outline=(210, 210, 210),
            width=1 * scale,
        )

        for _, node in group.iterrows():
            x, y = positions[str(node["Id"])]
            degree = float(node["Degree"])
            node_r = (7 + 13 * degree / max_degree) * scale
            draw.ellipse(
                (x - node_r, y - node_r, x + node_r, y + node_r),
                fill=color,
                outline=(38, 38, 38),
                width=max(1, scale),
            )

        if len(group) >= 5:
            label = f"C{community_id}"
            bbox = draw.textbbox((0, 0), label, font=label_font)
            draw.text(
                (cx - (bbox[2] - bbox[0]) / 2, cy + min(cell_w, cell_h) * 0.30),
                label,
                fill=(45, 45, 45),
                font=label_font,
            )

    draw.text((90 * scale, 42 * scale), "SASHTS Contact Network - Louvain Communities", fill=(20, 24, 28), font=title_font)
    draw.text(
        (90 * scale, 92 * scale),
        "Python-rendered Gephi export: node color = Louvain community, node size = degree, edge width = contact duration",
        fill=(78, 84, 91),
        font=subtitle_font,
    )
    draw.text(
        (90 * scale, (height - 70) * scale),
        f"{len(nodes)} nodes | {len(edges)} edges | {len(communities)} Louvain communities | Source: contact_network_louvain_colored.gexf",
        fill=(78, 84, 91),
        font=small_font,
    )

    output = canvas.resize((width, height), Image.Resampling.LANCZOS)
    output.save(GEPHI_DIR / "contact_network_louvain_colored.png")
    output.save(VIS_DIR / "gephi_louvain_community.png")
    export_selected_communities(nodes, edges)

    print("Saved:")
    print("  results/gephi/contact_network_louvain_colored.png")
    print("  visualizations/gephi_louvain_community.png")
    print("  results/gephi/louvain_selected_communities.png")
    print("  visualizations/louvain_selected_communities.png")


def export_selected_communities(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    community_sizes = nodes.groupby("Community Id").size().sort_values()
    smallest = int(community_sizes.index[0])
    largest_two = [int(idx) for idx in community_sizes.sort_values(ascending=False).index[:2]]
    selected_communities = [smallest, *largest_two]

    selected_nodes = nodes[nodes["Community Id"].isin(selected_communities)].copy()
    selected_ids = set(selected_nodes["Id"].astype(str))
    selected_edges = edges[
        edges["Source"].astype(str).isin(selected_ids) & edges["Target"].astype(str).isin(selected_ids)
    ].copy()

    width, height = 1800, 920
    scale = 2
    canvas = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(31 * scale, bold=True)
    subtitle_font = load_font(17 * scale)
    community_title_font = load_font(24 * scale, bold=True)
    community_subtitle_font = load_font(16 * scale)
    edge_font = load_font(12 * scale)

    draw.text((70 * scale, 36 * scale), "Selected Louvain Communities", fill=(20, 24, 28), font=title_font)
    draw.text(
        (70 * scale, 82 * scale),
        "Smallest community and two largest communities; edge labels show contact-duration weights",
        fill=(78, 84, 91),
        font=subtitle_font,
    )

    panels = [
        (smallest, "Smallest"),
        (largest_two[0], "Largest"),
        (largest_two[1], "2nd largest"),
    ]
    panel_w = (width * scale - 140 * scale) / 3
    top = 145 * scale
    panel_h = 720 * scale
    positions: dict[str, tuple[float, float]] = {}

    for panel_idx, (community_id, label) in enumerate(panels):
        x0 = 70 * scale + panel_idx * panel_w
        x1 = x0 + panel_w - 25 * scale
        y0 = top
        y1 = top + panel_h
        group = selected_nodes[selected_nodes["Community Id"] == community_id].sort_values("Id")
        color = hex_to_rgb(group.iloc[0]["Community Color"])
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2 + 45 * scale
        radius = min(x1 - x0, y1 - y0) * (0.34 if len(group) > 2 else 0.24)

        header = f"Community C{community_id}"
        subtitle = f"{label} | {len(group)} nodes"
        bbox = draw.textbbox((0, 0), header, font=community_title_font)
        draw.text(
            (cx - (bbox[2] - bbox[0]) / 2, y0),
            header,
            fill=(20, 24, 28),
            font=community_title_font,
        )
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=community_subtitle_font)
        draw.text(
            (cx - (subtitle_bbox[2] - subtitle_bbox[0]) / 2, y0 + 38 * scale),
            subtitle,
            fill=(87, 93, 99),
            font=community_subtitle_font,
        )

        for idx, (_, node) in enumerate(group.iterrows()):
            if len(group) == 1:
                x, y = cx, cy
            elif len(group) == 2:
                x = cx
                y = cy + (-95 if idx == 0 else 95) * scale
            else:
                angle = 2 * math.pi * idx / len(group) - math.pi / 2
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
            positions[str(node["Id"])] = (x, y)

        community_edges = selected_edges[
            selected_edges["Source"].astype(str).isin(set(group["Id"].astype(str)))
            & selected_edges["Target"].astype(str).isin(set(group["Id"].astype(str)))
        ]

        for _, edge in community_edges.iterrows():
            source, target = str(edge["Source"]), str(edge["Target"])
            x_a, y_a = positions[source]
            x_b, y_b = positions[target]
            draw.line((x_a, y_a, x_b, y_b), fill=(176, 181, 186), width=2 * scale)
            mid_x, mid_y = (x_a + x_b) / 2, (y_a + y_b) / 2
            weight_label = f"{int(edge['Weight']):,}"
            text_box = draw.textbbox((0, 0), weight_label, font=edge_font)
            pad = 3 * scale
            draw.rounded_rectangle(
                (
                    mid_x - (text_box[2] - text_box[0]) / 2 - pad,
                    mid_y - (text_box[3] - text_box[1]) / 2 - pad,
                    mid_x + (text_box[2] - text_box[0]) / 2 + pad,
                    mid_y + (text_box[3] - text_box[1]) / 2 + pad,
                ),
                radius=4 * scale,
                fill=(255, 255, 255),
                outline=(222, 226, 230),
            )
            draw.text(
                (mid_x - (text_box[2] - text_box[0]) / 2, mid_y - (text_box[3] - text_box[1]) / 2),
                weight_label,
                fill=(65, 70, 75),
                font=edge_font,
            )

        for _, node in group.iterrows():
            x, y = positions[str(node["Id"])]
            node_r = 28 * scale
            draw.ellipse(
                (x - node_r, y - node_r, x + node_r, y + node_r),
                fill=color,
                outline=(38, 38, 38),
                width=2 * scale,
            )

    output = canvas.resize((width, height), Image.Resampling.LANCZOS)
    output.save(GEPHI_DIR / "louvain_selected_communities.png")
    output.save(VIS_DIR / "louvain_selected_communities.png")


if __name__ == "__main__":
    main()
