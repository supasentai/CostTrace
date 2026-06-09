from __future__ import annotations

import colorsys
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualizations"
GEPHI_DIR = ROOT / "results" / "gephi"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "phase05.log"
GEXF_NS = "http://www.gexf.net/1.2draft"
VIZ_NS = "http://www.gexf.net/1.2draft/viz"

ET.register_namespace("", GEXF_NS)
ET.register_namespace("viz", VIZ_NS)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def gephi_title(name: str) -> str:
    return name.replace("_", " ").title()


def community_palette(community_ids: list[int]) -> dict[int, tuple[int, int, int, str]]:
    palette = {}
    for idx, community_id in enumerate(sorted(community_ids)):
        hue = (idx * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.88)
        rgb = tuple(int(round(channel * 255)) for channel in (red, green, blue))
        palette[int(community_id)] = (*rgb, "#{:02X}{:02X}{:02X}".format(*rgb))
    return palette


def write_colored_gexf(node_attrs: pd.DataFrame, palette: dict[int, tuple[int, int, int, str]]) -> None:
    source_gexf = GEPHI_DIR / "contact_network.gexf"
    output_gexf = GEPHI_DIR / "contact_network_louvain_colored.gexf"

    tree = ET.parse(source_gexf)
    root = tree.getroot()
    graph = root.find(f"{{{GEXF_NS}}}graph")
    node_attributes = graph.find(f"{{{GEXF_NS}}}attributes[@class='node']")

    existing_ids = [
        int(attr.attrib["id"])
        for attr in node_attributes.findall(f"{{{GEXF_NS}}}attribute")
        if attr.attrib.get("id", "").isdigit()
    ]
    next_id = max(existing_ids, default=-1) + 1
    community_attr_id = str(next_id)
    color_attr_id = str(next_id + 1)

    ET.SubElement(
        node_attributes,
        f"{{{GEXF_NS}}}attribute",
        {"id": community_attr_id, "title": "Louvain Community Id", "type": "integer"},
    )
    ET.SubElement(
        node_attributes,
        f"{{{GEXF_NS}}}attribute",
        {"id": color_attr_id, "title": "Community Color", "type": "string"},
    )

    lookup = node_attrs.set_index("Id")[["Louvain Community Id", "Community Color"]].to_dict("index")
    for node in graph.find(f"{{{GEXF_NS}}}nodes").findall(f"{{{GEXF_NS}}}node"):
        node_id = node.attrib["id"]
        record = lookup.get(node_id)
        if record is None or pd.isna(record["Louvain Community Id"]):
            continue

        community_id = int(record["Louvain Community Id"])
        red, green, blue, hex_color = palette[community_id]
        attvalues = node.find(f"{{{GEXF_NS}}}attvalues")
        if attvalues is None:
            attvalues = ET.SubElement(node, f"{{{GEXF_NS}}}attvalues")
        ET.SubElement(attvalues, f"{{{GEXF_NS}}}attvalue", {"for": community_attr_id, "value": str(community_id)})
        ET.SubElement(attvalues, f"{{{GEXF_NS}}}attvalue", {"for": color_attr_id, "value": hex_color})
        ET.SubElement(node, f"{{{VIZ_NS}}}color", {"r": str(red), "g": str(green), "b": str(blue), "a": "1.0"})

    tree.write(output_gexf, encoding="utf-8", xml_declaration=True)


def main() -> None:
    setup_logging()
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    GEPHI_DIR.mkdir(parents=True, exist_ok=True)

    scores_path = ROOT / "results" / "metrics" / "node_scores.csv"
    community_path = ROOT / "results" / "metrics" / "community_assignments.csv"
    edges_path = ROOT / "data" / "processed" / "edgelist.csv"

    scores_df = pd.read_csv(scores_path)
    comm_df = pd.read_csv(community_path)
    node_attrs = scores_df.merge(comm_df, on="node_id", how="left", suffixes=("", "_community"))
    node_attrs = node_attrs.rename(columns={col: gephi_title(col) for col in node_attrs.columns})
    node_attrs = node_attrs.rename(columns={"Node Id": "Id"})

    # Tao moi palette va cot mau de Gephi co the to mau node theo Louvain community.
    community_ids = sorted(node_attrs["Louvain Community Id"].dropna().astype(int).unique().tolist())
    palette = community_palette(community_ids)
    node_attrs["Community Color"] = node_attrs["Louvain Community Id"].map(
        lambda value: palette[int(value)][3] if pd.notna(value) else "#999999"
    )
    palette_df = pd.DataFrame(
        [
            {
                "Louvain Community Id": community_id,
                "Community Color": hex_color,
                "R": red,
                "G": green,
                "B": blue,
            }
            for community_id, (red, green, blue, hex_color) in palette.items()
        ]
    )
    palette_df.to_csv(VIS_DIR / "gephi_louvain_palette.csv", index=False)
    node_attrs.to_csv(VIS_DIR / "gephi_nodes.csv", index=False)

    edges_df = pd.read_csv(edges_path)
    gephi_edges = edges_df.rename(columns={"source": "Source", "target": "Target", "weight": "Weight"})
    keep_cols = [col for col in ["Source", "Target", "Weight", "n_contacts", "transmission"] if col in gephi_edges]
    gephi_edges = gephi_edges[keep_cols]
    gephi_edges["Type"] = "Undirected"
    gephi_edges = gephi_edges.rename(columns={"n_contacts": "Contact Count", "transmission": "Transmission"})
    gephi_edges.to_csv(VIS_DIR / "gephi_edges.csv", index=False)

    # Tao moi GEXF da nhung viz:color de Gephi mo len co mau community theo Louvain.
    write_colored_gexf(node_attrs, palette)

    logging.info("Exported Gephi CSV/GEXF files to %s and %s", VIS_DIR, GEPHI_DIR)
    print("Gephi files exported:")
    print("  visualizations/gephi_nodes.csv")
    print("  visualizations/gephi_edges.csv")
    print("  visualizations/gephi_louvain_palette.csv")
    print("  results/gephi/contact_network_louvain_colored.gexf")
    print("\nGEPHI INSTRUCTIONS:")
    print("1. Open results/gephi/contact_network_louvain_colored.gexf")
    print("2. Layout: ForceAtlas2 (run about 2000 steps)")
    print("3. Appearance -> Nodes -> Color -> Partition -> Louvain Community Id")
    print("4. Appearance -> Nodes -> Size -> Ranking -> degree / weighted_degree_sec")
    print("5. Export: File -> Export -> SVG/PNG (2000x2000px)")


if __name__ == "__main__":
    main()
