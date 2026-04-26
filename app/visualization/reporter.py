"""PDF report generation."""
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def calculate_area(water_mask: np.ndarray, transform) -> float:
    """Calculates total flooded area in hectares based on pixel dimensions."""
    water_pixels = np.sum(water_mask == 1)
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    area_per_pixel_sqm = pixel_width * pixel_height
    return (water_pixels * area_per_pixel_sqm) / 10000.0


def generate_flood_report(location: str, start_date: str, end_date: str,
                          flood_area_ha: float, output_html_path: str):
    """Generates a lightweight HTML summary report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>FloodLLM Analysis Report - {location}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }}
            .summary-card {{ background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-top: 20px; }}
            .metric {{ font-size: 1.2em; margin: 10px 0; }}
            .highlight {{ font-weight: bold; color: #d9534f; font-size: 1.5em; }}
        </style>
    </head>
    <body>
        <h1>FloodLLM Detection Report</h1>
        <div class="summary-card">
            <h2>Analysis Summary</h2>
            <div class="metric"><strong>Location:</strong> {location}</div>
            <div class="metric"><strong>Period:</strong> {start_date} to {end_date}</div>
            <div class="metric"><strong>Flooded Area:</strong> <span class="highlight">{flood_area_ha:,.2f} ha</span></div>
        </div>
        <p style="margin-top:40px; color:#777; text-align:center;">Generated on {timestamp}</p>
    </body>
    </html>
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ..utils.config import settings


class ReportGenerator:
    """Generate PDF flood assessment reports."""

    def __init__(self, job_id: Optional[str] = None):
        """Initialize report generator."""
        if job_id:
            self.job_dir = settings.get_job_dir(job_id)
            self.output_dir = self.job_dir / "reports"
        else:
            self.output_dir = settings.output_dir / "reports"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None

    def generate_report(
        self,
        report_data: Dict[str, Any],
        job_id: str
    ) -> str:
        """
        Generate PDF flood assessment report.
        """
        if job_id and job_id != self.job_id:
            self.job_id = job_id
            self.job_dir = settings.get_job_dir(job_id)
            self.output_dir = self.job_dir / "reports"

        pdf_path = self.output_dir / f"flood_report_{job_id}.pdf"

        if not REPORTLAB_AVAILABLE:
            # Generate HTML report as fallback
            return self._generate_html_report(report_data, job_id)

        try:
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )

            # Build report content
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                leading=22,
                alignment=TA_CENTER,
                spaceAfter=30
            )
            story.append(Paragraph("FLOOD ASSESSMENT REPORT", title_style))

            # Metadata table
            meta_table = Table([
                ['Location:', report_data.get('location', 'Unknown')],
                ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M')],
                ['Period Covered:', report_data.get('date_range', 'N/A')],
                ['Report ID:', job_id]
            ], colWidths=[1.5*inch, 4*inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 20))

            # Executive Summary
            story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['Heading2']))

            summary_text = self._generate_summary(report_data)
            story.append(Paragraph(summary_text, self.styles['Normal']))
            story.append(Spacer(1, 15))

            # Key Statistics
            story.append(Paragraph("KEY STATISTICS", self.styles['Heading2']))

            stats_data = [
                ['Metric', 'Value'],
                ['Flood Area', f"{report_data.get('flood_area_km2', 0):.2f} km²"],
                ['Affected Buildings', f"~{report_data.get('affected_buildings', 0)}"],
                ['Affected Roads', f"~{report_data.get('affected_roads_km', 0):.1f} km"],
                ['Agricultural Land', f"~{report_data.get('agricultural_km2', 0):.2f} km²"],
                ['Rainfall (period)', f"{report_data.get('rainfall_mm', 0):.1f} mm"]
            ]

            stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke)
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 20))

            # Risk Assessment
            if 'risk_assessment' in report_data:
                story.append(Paragraph("RISK ASSESSMENT", self.styles['Heading2']))

                risk = report_data['risk_assessment']
                risk_text = f"""
                <b>Overall Risk Level:</b> {risk.get('level', 'Unknown').upper()}<br/><br/>
                High Risk Area: {risk.get('high_risk_pct', 0):.1f}%<br/>
                Moderate Risk Area: {risk.get('moderate_risk_pct', 0):.1f}%<br/>
                Low Risk Area: {risk.get('low_risk_pct', 0):.1f}%
                """
                story.append(Paragraph(risk_text, self.styles['Normal']))
                story.append(Spacer(1, 15))

            # Recommendations
            story.append(Paragraph("RECOMMENDATIONS", self.styles['Heading2']))

            recommendations = report_data.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
                story.append(Spacer(1, 5))

            # LLM-generated narrative (if available)
            if 'narrative' in report_data and report_data['narrative']:
                story.append(Spacer(1, 20))
                story.append(Paragraph("DETAILED ASSESSMENT", self.styles['Heading2']))

                # Split narrative into paragraphs
                paragraphs = report_data['narrative'].split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), self.styles['Normal']))
                        story.append(Spacer(1, 10))

            # Footer
            story.append(Spacer(1, 30))
            footer = Paragraph(
                "<i>Generated by FloodLLM - Automated Flood Monitoring System</i>",
                ParagraphStyle('Footer', parent=self.styles['Normal'], alignment=TA_CENTER)
            )
            story.append(footer)

            # Build PDF
            doc.build(story)

            return str(pdf_path)

        except Exception as e:
            print(f"PDF generation error: {e}")
            return self._generate_html_report(report_data, job_id)

    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary text."""
        flood_area = data.get('flood_area_km2', 0)

        if flood_area > 100:
            severity = "SEVERE"
        elif flood_area > 50:
            severity = "MODERATE"
        elif flood_area > 10:
            severity = "MINOR"
        else:
            severity = "LOCALIZED"

        location = data.get('location', 'the affected area')

        return f"""
        Satellite-based analysis has detected <b>{flood_area:.2f} km²</b> of flooded area in {location}.
        The flood severity is classified as <b>{severity}</b>. Immediate attention is recommended for
        low-lying areas and communities near water bodies. Approximately {data.get('affected_buildings', 0)}
        buildings and {data.get('affected_roads_km', 0):.1f} km of roads may be affected.
        """

    def _generate_html_report(
        self,
        report_data: Dict[str, Any],
        job_id: str
    ) -> str:
        """Generate HTML report as fallback."""
        html_path = self.output_dir / f"flood_report_{job_id}.html"
        
        risk = report_data.get('risk_assessment', {})
        risk_level = risk.get('level', 'Unknown')
        
        # Determine color for risk level
        risk_color = "#e74c3c" if risk_level == "High" else "#f39c12" if risk_level == "Medium" else "#27ae60"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Flood Assessment Report - {job_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; max-width: 900px; margin-left: auto; margin-right: auto; }}
        header {{ text-align: center; border-bottom: 3px solid #2980b9; padding-bottom: 20px; margin-bottom: 30px; }}
        h1 {{ color: #2c3e50; margin: 0; }}
        h2 {{ color: #2980b9; border-bottom: 1px solid #eee; padding-bottom: 8px; margin-top: 30px; }}
        .meta-container {{ display: flex; justify-content: space-between; background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 5px solid #2980b9; }}
        .meta-item {{ flex: 1; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #2980b9; display: block; }}
        .stat-label {{ font-size: 0.9em; color: #7f8c8d; }}
        .risk-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; background-color: {risk_color}; }}
        .recommendation {{ background: #fff8e1; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; border-radius: 0 4px 4px 0; }}
        .summary-box {{ background: #eef7ff; padding: 20px; border-radius: 8px; font-style: italic; }}
        footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <header>
        <h1>FLOOD ASSESSMENT REPORT</h1>
        <p>Generated by FloodLLM Autonomous Monitoring</p>
    </header>

    <div class="meta-container">
        <div class="meta-item">
            <strong>📍 Location:</strong> {report_data.get('location', 'Unknown')}<br>
            <strong>📅 Period:</strong> {report_data.get('date_range', 'N/A')}
        </div>
        <div class="meta-item" style="text-align: right;">
            <strong>🆔 Job ID:</strong> {job_id}<br>
            <strong>⏰ Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>

    <h2>Executive Summary</h2>
    <div class="summary-box">
        {self._generate_summary(report_data)}
    </div>

    <h2>Risk Analysis</h2>
    <p>Overall Risk Level: <span class="risk-badge">{risk_level.upper()}</span></p>
    <div class="stats-grid">
        <div class="stat-card">
            <span class="stat-value">{risk.get('high_risk_pct', 0):.1f}%</span>
            <span class="stat-label">High Risk Area</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">{risk.get('moderate_risk_pct', 0):.1f}%</span>
            <span class="stat-label">Moderate Risk Area</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">{risk.get('low_risk_pct', 0):.1f}%</span>
            <span class="stat-label">Low Risk Area</span>
        </div>
    </div>

    <h2>Impact Statistics</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <span class="stat-value">🌊 {report_data.get('flood_area_km2', 0):.2f}</span>
            <span class="stat-label">Flood Area (km²)</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">🏠 {report_data.get('affected_buildings', 0)}</span>
            <span class="stat-label">Affected Buildings</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">🛣️ {report_data.get('affected_roads_km', 0):.1f}</span>
            <span class="stat-label">Affected Roads (km)</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">🌧️ {report_data.get('rainfall_mm', 0):.1f}</span>
            <span class="stat-label">Rainfall (mm)</span>
        </div>
    </div>

    <h2>Recommendations</h2>
    {self._format_recommendations_html(report_data.get('recommendations', []))}

    <footer>
        &copy; 2026 FloodLLM System. All geospatial data derived from Sentinel-1, Sentinel-2, and GPM-IMERG.
    </footer>
</body>
</html>
"""

        with open(html_path, 'w') as f:
            f.write(html_content)

        return str(html_path)

    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations for HTML."""
        html = ""
        for i, rec in enumerate(recommendations, 1):
            html += f'<div class="recommendation">{i}. {rec}</div>'
        return html
