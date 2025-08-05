# population_metric_exporter.py
# Usage Example:
# >>> exporter = MetricExporter(population, output_dir=Path("Output/IS1/"))
# >>> exporter.export_all()

import csv
from calcium_activity_characterization.logger import logger
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from calcium_activity_characterization.data.populations import Population
from calcium_activity_characterization.analysis.metrics import Distribution




class MetricExporter:
    """
    Exports population-level metric distributions (cell, sequential event, global event)
    as CSV and PDF report.

    Args:
        population (Population): The analyzed population object.
        output_dir (Path): Directory to save exported files.
    """

    def __init__(self, population: Population, output_dir: Path) -> None:
        self.population = population
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self) -> None:
        """Export CSVs and PDF summary."""
        self.export_distributions_to_csv()
        self.export_pdf_summary()

    def export_distributions_to_csv(self) -> None:
        """Export all distributions as summary CSVs."""
        try:
            self._save_dist_dict(self.population.cell_metrics_distributions, "cell_metrics.csv")
            self._save_dist_dict(self.population.seq_event_metrics_distributions, "seq_event_metrics.csv")
            self._save_dist_dict(self.population.glob_event_metrics_distributions, "global_event_metrics.csv")
        except Exception as e:
            logger.error(f"Failed to export distribution CSVs: {e}")

    def _save_dist_dict(self, dists: dict[str, Distribution], filename: str) -> None:
        """Save one dictionary of metric -> Distribution to a CSV."""
        try:
            path = self.output_dir / filename
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max", "count"])
                writer.writeheader()
                for metric, dist in dists.items():
                    row = {"metric": metric, **dist.as_dict()}
                    writer.writerow(row)
            logger.info(f"Saved metric summary to {path}")
        except Exception as e:
            logger.error(f"Failed to write CSV {filename}: {e}")

    def export_pdf_summary(self) -> None:
        """Generate a PDF report with all histograms and distribution stats."""
        try:
            pdf_path = self.output_dir / "metrics_report.pdf"
            with PdfPages(pdf_path) as pdf:
                for category, dists in [
                    ("Cell Metrics", self.population.cell_metrics_distributions),
                    ("Sequential Event Metrics", self.population.seq_event_metrics_distributions),
                    ("Global Event Metrics", self.population.glob_event_metrics_distributions),
                ]:
                    for metric, dist in dists.items():
                        if dist.count == 0:
                            continue
                        fig = dist.plot_histogram(title=f"{category} - {metric}", xlabel=metric)
                        pdf.savefig(fig)
                        plt.close(fig)
            logger.info(f"Saved PDF report to {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to generate PDF summary: {e}")