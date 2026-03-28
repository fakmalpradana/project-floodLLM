import os
import numpy as np
import rasterio
from datetime import datetime

def calculate_area(water_mask: np.ndarray, transform: rasterio.Affine) -> float:
    """
    Calculates the total flooded area in hectares based on pixel dimensions.
    
    Args:
        water_mask (np.ndarray): Binary mask where 1 = water.
        transform (rasterio.Affine): Affine transform from raster metadata containing pixel resolution.
        
    Returns:
        float: Total flooded area in hectares.
    """
    # Count number of valid water pixels
    water_pixels = np.sum(water_mask == 1)
    
    # Extract pixel size from Affine transform 
    # (Assumes CRS map units are in meters, e.g., UTM projection)
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    
    # Calculate area per pixel in square meters
    area_per_pixel_sqm = pixel_width * pixel_height
    
    # Convert total square meters to hectares (1 ha = 10,000 m^2)
    total_area_ha = (water_pixels * area_per_pixel_sqm) / 10000.0
    return total_area_ha

def generate_flood_report(location: str, start_date: str, end_date: str, 
                          flood_area_ha: float, output_html_path: str):
    """
    Generates a lightweight, structured HTML summary report for the flood analysis.
    
    Args:
        location (str): Name of the analyzed area.
        start_date (str): Start date of the temporal window.
        end_date (str): End date of the temporal window.
        flood_area_ha (float): Calculated flood extent in hectares.
        output_html_path (str): File path to save the generated HTML report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>FloodLLM Analysis Report - {location}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }}
            .summary-card {{ background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-top: 20px; }}
            .metric {{ font-size: 1.2em; margin: 10px 0; }}
            .highlight {{ font-weight: bold; color: #d9534f; font-size: 1.5em; }}
            .footer {{ margin-top: 40px; font-size: 0.9em; color: #777; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>🌊 FloodLLM Detection Report</h1>
        
        <div class="summary-card">
            <h2>Analysis Summary</h2>
            <div class="metric"><strong>Location:</strong> {location}</div>
            <div class="metric"><strong>Analysis Period:</strong> {start_date} to {end_date}</div>
            <div class="metric"><strong>Total Flooded Area Detected:</strong> <span class="highlight">{flood_area_ha:,.2f} hectares</span></div>
        </div>
        
        <div class="footer">
            <p>Report generated automatically by FloodLLM System on {timestamp}</p>
        </div>
    </body>
    </html>
    """
    
    # Safely create parent directories if they don't exist
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report generated successfully: {output_html_path}")
