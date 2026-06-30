import os
import json
import csv
import logging
from jinja2 import Template

# PDF reporting
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DatasetReporter:
    def __init__(self, metrics: dict, output_dir: str = "reports"):
        self.metrics = metrics
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_csv(self):
        """
        Exports the most common words and general parameters to CSV files.
        """
        csv_path = os.path.join(self.output_dir, "word_frequencies.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Word", "Frequency"])
            for word, freq in self.metrics.get("most_common_words", []):
                writer.writerow([word, freq])
        logger.info(f"Exported word frequencies to CSV: {csv_path}")

        # Summary parameters CSV
        summary_path = os.path.join(self.output_dir, "summary_metrics.csv")
        with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value"])
            for key in ["total_images", "mean_width", "mean_height", "vocab_size", "total_words", "mean_caption_length"]:
                writer.writerow([key, self.metrics.get(key, "N/A")])
        logger.info(f"Exported summary metrics to CSV: {summary_path}")

    def export_html(self, plots_dir: str = "../plots"):
        """
        Compiles an interactive HTML report summary page.
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Explorer Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; color: #333; }
                .container { max-width: 1000px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin: auto; }
                h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
                h2 { color: #16a085; margin-top: 30px; }
                .metric-card { display: inline-block; background: #ecf0f1; padding: 15px 25px; border-radius: 5px; margin-right: 15px; margin-bottom: 15px; min-width: 150px; text-align: center; }
                .metric-val { font-size: 24px; font-weight: bold; color: #2980b9; }
                .plot-img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: 15px; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #34495e; color: white; }
                tr:nth-child(even){background-color: #f9f9f9;}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Dataset Analysis Report</h1>
                <p>Generated automatically by the AI/ML Dataset Explorer tool.</p>
                
                <h2>General Statistics</h2>
                <div>
                    <div class="metric-card">
                        <div>Total Images</div>
                        <div class="metric-val">{{ metrics.total_images }}</div>
                    </div>
                    <div class="metric-card">
                        <div>Vocabulary Size</div>
                        <div class="metric-val">{{ metrics.vocab_size }}</div>
                    </div>
                    <div class="metric-card">
                        <div>Mean Image Size</div>
                        <div class="metric-val">{{ metrics.mean_width | int }}x{{ metrics.mean_height | int }}</div>
                    </div>
                    <div class="metric-card">
                        <div>Mean Caption Length</div>
                        <div class="metric-val">{{ "%.2f"|format(metrics.mean_caption_length) }} words</div>
                    </div>
                </div>

                <h2>Visual Distributions</h2>
                <div style="text-align: center;">
                    <h3>Word Frequency Chart</h3>
                    <img class="plot-img" src="{{ plots_dir }}/word_frequencies.png" alt="Word Frequency distribution">
                    
                    <h3>Caption Length Histogram</h3>
                    <img class="plot-img" src="{{ plots_dir }}/caption_lengths.png" alt="Caption lengths">
                </div>

                <h2>Vocabulary Word Frequency Table</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Word</th>
                            <th>Occurrences</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for word, count in metrics.most_common_words %}
                        <tr>
                            <td>{{ loop.index }}</td>
                            <td>{{ word }}</td>
                            <td>{{ count }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        template = Template(html_template)
        rendered_html = template.render(metrics=self.metrics, plots_dir=plots_dir)
        
        html_path = os.path.join(self.output_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        logger.info(f"Generated HTML report at: {html_path}")

    def export_pdf(self, plots_dir: str = "plots"):
        """
        Compiles a formal PDF report including metrics and charts.
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab package not found. Skipping PDF generation.")
            return

        pdf_path = os.path.join(self.output_dir, "report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Heading1"],
            fontSize=22,
            textColor=colors.HexColor("#2C3E50"),
            spaceAfter=15
        )
        subtitle_style = ParagraphStyle(
            name="SubTitleStyle",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#16A085"),
            spaceAfter=10,
            spaceBefore=15
        )
        body_style = styles["Normal"]

        story = []
        
        # Header / Title
        story.append(Paragraph("Dataset Exploration Analysis Report", title_style))
        story.append(Paragraph("This PDF report summarizes image resolution distributions, caption counts, and word histograms.", body_style))
        story.append(Spacer(1, 15))

        # Metrics Table
        table_data = [
            ["Metric Parameter", "Value"],
            ["Total Images", str(self.metrics.get("total_images"))],
            ["Vocabulary Size", str(self.metrics.get("vocab_size"))],
            ["Mean Dimensions", f"{int(self.metrics.get('mean_width', 0))}x{int(self.metrics.get('mean_height', 0))}"],
            ["Mean Caption Length", f"{self.metrics.get('mean_caption_length', 0):.2f} words"],
            ["Max Caption Length", str(self.metrics.get("max_caption_length"))]
        ]
        
        t = Table(table_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#34495E")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#ECF0F1")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#BDC3C7")),
            ('FONTSIZE', (0,0), (-1,-1), 10),
        ]))
        
        story.append(Paragraph("Dataset Metrics Overview", subtitle_style))
        story.append(t)
        story.append(Spacer(1, 15))

        # Charts / Images (if generated)
        word_plot_path = os.path.join(plots_dir, "word_frequencies.png")
        if os.path.exists(word_plot_path):
            story.append(Paragraph("Word Frequency Distribution Chart", subtitle_style))
            # Rescale image to fit page widths (approx 400x200)
            story.append(RLImage(word_plot_path, width=400, height=200))
            story.append(Spacer(1, 15))
            
        len_plot_path = os.path.join(plots_dir, "caption_lengths.png")
        if os.path.exists(len_plot_path):
            story.append(Paragraph("Caption Length Frequency Distribution", subtitle_style))
            story.append(RLImage(len_plot_path, width=400, height=200))
            story.append(Spacer(1, 15))

        doc.build(story)
        logger.info(f"Generated PDF report at: {pdf_path}")
