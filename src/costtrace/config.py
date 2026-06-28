from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_sashts(self) -> Path:
        return self.data / "raw" / "sashts"

    @property
    def processed_sashts(self) -> Path:
        return self.data / "processed" / "sashts"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def metrics(self) -> Path:
        return self.results / "metrics"

    @property
    def model_results(self) -> Path:
        return self.results / "model"

    @property
    def intervention(self) -> Path:
        return self.results / "intervention"

    @property
    def audit(self) -> Path:
        return self.results / "audit"

    @property
    def gephi(self) -> Path:
        return self.results / "gephi"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def raw_contact_network_canonical(self) -> Path:
        return self.raw_sashts / "sashts_contact_network.csv"

    @property
    def raw_metadata_canonical(self) -> Path:
        return self.raw_sashts / "sashts_metadata.csv"

    @property
    def processed_edges_clean_canonical(self) -> Path:
        return self.processed_sashts / "edges_clean.csv"

    @property
    def processed_metadata_canonical(self) -> Path:
        return self.processed_sashts / "metadata_clean.csv"

    @property
    def processed_graph_canonical(self) -> Path:
        return self.processed_sashts / "graph.pkl"

    @property
    def processed_nodelist_canonical(self) -> Path:
        return self.processed_sashts / "nodelist.csv"

    @property
    def processed_edgelist_canonical(self) -> Path:
        return self.processed_sashts / "edgelist.csv"

    @property
    def eda_summary_canonical(self) -> Path:
        return self.processed_sashts / "eda_summary.json"

    @property
    def raw_contact_network(self) -> Path:
        return prefer_existing(
            self.raw_contact_network_canonical,
            self.data / "raw" / "sashts_contact_network.csv",
        )

    @property
    def raw_metadata(self) -> Path:
        return prefer_existing(
            self.raw_metadata_canonical,
            self.data / "raw" / "sashts_metadata.csv",
        )

    @property
    def processed_edges_clean(self) -> Path:
        return prefer_existing(
            self.processed_edges_clean_canonical,
            self.data / "processed" / "edges_clean.csv",
        )

    @property
    def processed_metadata(self) -> Path:
        return prefer_existing(
            self.processed_metadata_canonical,
            self.data / "processed" / "metadata_clean.csv",
        )

    @property
    def processed_graph(self) -> Path:
        return prefer_existing(
            self.processed_graph_canonical,
            self.data / "processed" / "graph.pkl",
        )

    @property
    def processed_nodelist(self) -> Path:
        return prefer_existing(
            self.processed_nodelist_canonical,
            self.data / "processed" / "nodelist.csv",
        )

    @property
    def processed_edgelist(self) -> Path:
        return prefer_existing(
            self.processed_edgelist_canonical,
            self.data / "processed" / "edgelist.csv",
        )

    @property
    def eda_summary(self) -> Path:
        return prefer_existing(
            self.eda_summary_canonical,
            self.data / "processed" / "eda_summary.json",
        )


def prefer_existing(canonical: Path, legacy: Path | None = None) -> Path:
    if canonical.exists():
        return canonical
    if legacy is not None and legacy.exists():
        return legacy
    return canonical


def require_existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


PATHS = ProjectPaths()
