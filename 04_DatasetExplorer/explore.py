import argparse
import logging
import yaml
import os

from src.analyzer import DatasetAnalyzer
from src.plotter import generate_plots
from src.reporter import DatasetReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Dataset Analysis & Statistics Tool")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Overrides config dataset path")
    parser.add_argument("--format", type=str, default=None, choices=["custom", "coco", "oxford"], help="Overrides config dataset format")
    parser.add_argument("--reports_dir", type=str, default=None, help="Overrides config reports directory")
    parser.add_argument("--plots_dir", type=str, default=None, help="Overrides config plots directory")
    args = parser.parse_args()

    # Load configuration settings
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Overrides
    dataset_path = args.dataset_dir if args.dataset_dir else config["dataset"]["path"]
    dataset_format = args.format if args.format else config["dataset"]["format"]
    reports_dir = args.reports_dir if args.reports_dir else config["outputs"]["reports_dir"]
    plots_dir = args.plots_dir if args.plots_dir else config["outputs"]["plots_dir"]

    # Run analysis
    analyzer = DatasetAnalyzer(dataset_path=dataset_path, dataset_format=dataset_format)
    metrics = analyzer.analyze()

    if metrics.get("status") == "empty":
        logger.error(f"No valid images/captions found in {dataset_path}.")
        return

    # Generate charts
    logger.info("Generating statistical charts and plots...")
    generate_plots(metrics, output_dir=plots_dir)

    # Write reports
    logger.info("Writing statistics reports (CSV, HTML, PDF)...")
    reporter = DatasetReporter(metrics, output_dir=reports_dir)
    reporter.export_csv()
    
    # HTML template needs relative path to plots
    rel_plots_dir = os.path.relpath(plots_dir, start=reports_dir)
    reporter.export_html(plots_dir=rel_plots_dir)
    
    # PDF generation
    reporter.export_pdf(plots_dir=plots_dir)

    logger.info(f"Analysis successfully completed. Reports saved in: {reports_dir}")

if __name__ == "__main__":
    main()
