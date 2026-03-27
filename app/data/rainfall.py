"""Rainfall data download (GPM, CHIRPS)."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

import httpx

from ..utils.config import settings


class RainfallDownloader:
    """Download rainfall data from GPM and CHIRPS."""

    def __init__(self):
        """Initialize rainfall downloader."""
        self.data_dir = settings.data_dir / "rainfall"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def download_gpm(
        self,
        bbox: tuple,
        date_start: str,
        date_end: str
    ) -> Optional[Dict[str, Any]]:
        """
        Download GPM IMERG rainfall data.

        Uses NASA Giovanni API for accessible download.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            date_start: Start date
            date_end: End date

        Returns: Rainfall data info or None
        """
        date_start, date_end = self._parse_dates(date_start, date_end)

        try:
            # NASA Giovanni API endpoint for GPM
            # Using the GPM_L3_IMERGDLQ_V06 product (daily precipitation)
            base_url = "https://giovanni.gsfc.nasa.gov/giovanni/automation/serviceresponse"

            params = {
                'dataTool': 'mapImage',
                'version': '12.0',
                'dataSetId': 'GPM_L3_IMERGDLQ_V06',
                'parameter': 'precipitation',
                'startDate': date_start.strftime('%Y-%m-%dT00:00:00Z'),
                'endDate': date_end.strftime('%Y-%m-%dT23:59:59Z'),
                'bbox': f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # lat1,lon1,lat2,lon2
                'responseType': 'json'
            }

            # Note: This requires authentication for full access
            # For demo purposes, we'll use a simpler approach

            return await self._get_gpm_climatology(bbox, date_start, date_end)

        except Exception as e:
            print(f"GPM download error: {e}")
            return None

    async def _get_gpm_climatology(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime
    ) -> Dict[str, Any]:
        """
        Get GPM rainfall estimate using simplified approach.

        For MVP, we use a climatological estimate based on location and season.
        This will be replaced with actual API calls in production.
        """
        # Calculate approximate rainfall based on typical monsoon patterns
        # This is a placeholder - real implementation would use actual GPM data

        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # Simple seasonal model (tropical regions)
        day_of_year = date_start.timetuple().tm_yday

        # Wet season (Oct-Apr for Southern Hemisphere, May-Sep for Northern)
        if center_lat < 0:  # Southern hemisphere
            wet_season = (day_of_year < 90) or (day_of_year > 270)
        else:  # Northern hemisphere
            wet_season = (90 <= day_of_year <= 270)

        # Estimate daily rainfall (mm/day)
        if wet_season:
            daily_avg = 15 + (center_lat ** 2) * 0.1  # Higher near equator
        else:
            daily_avg = 5 + (center_lat ** 2) * 0.05

        days = (date_end - date_start).days + 1
        total_rainfall = daily_avg * days

        return {
            'source': 'GPM IMERG (estimated)',
            'bbox': bbox,
            'date_start': date_start.isoformat(),
            'date_end': date_end.isoformat(),
            'daily_avg_mm': round(daily_avg, 2),
            'total_mm': round(total_rainfall, 2),
            'days': days,
            'warning': 'Using climatological estimate - connect NASA API for real data'
        }

    async def download_chirps(
        self,
        bbox: tuple,
        date_start: str,
        date_end: str
    ) -> Optional[Dict[str, Any]]:
        """
        Download CHIRPS rainfall data.

        CHIRPS provides 0.05° resolution rainfall estimates.
        """
        date_start, date_end = self._parse_dates(date_start, date_end)

        try:
            # CHIRPS data via Google Earth Engine (preferred)
            # or direct FTP download

            # For MVP, provide simplified estimate
            return await self._get_chirps_estimate(bbox, date_start, date_end)

        except Exception as e:
            print(f"CHIRPS download error: {e}")
            return None

    async def _get_chirps_estimate(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime
    ) -> Dict[str, Any]:
        """Get CHIRPS-style rainfall estimate."""
        # Placeholder implementation
        min_lon, min_lat, max_lon, max_lat = bbox

        # CHIRPS-like estimate based on typical patterns
        area_km2 = self._calc_area_km2(bbox)

        # Assume moderate rainfall for demonstration
        daily_avg = 10  # mm/day
        days = (date_end - date_start).days + 1
        total_mm = daily_avg * days

        return {
            'source': 'CHIRPS (estimated)',
            'bbox': bbox,
            'date_start': date_start.isoformat(),
            'date_end': date_end.isoformat(),
            'daily_avg_mm': daily_avg,
            'total_mm': total_mm,
            'area_km2': area_km2,
            'warning': 'Connect to actual CHIRPS API for production use'
        }

    def _calc_area_km2(self, bbox: tuple) -> float:
        """Calculate area of bounding box in km²."""
        min_lon, min_lat, max_lon, max_lat = bbox
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * abs(((min_lat + max_lat) / 2) * 3.14159 / 180)

        width = (max_lon - min_lon) * km_per_deg_lon
        height = (max_lat - min_lat) * km_per_deg_lat

        return width * height

    def _parse_dates(self, date_start: str, date_end: str) -> tuple:
        """Parse date strings."""
        now = datetime.utcnow()

        if date_start in ['last 7 days', 'past week']:
            date_start = now - timedelta(days=7)
        elif date_start in ['last 14 days']:
            date_start = now - timedelta(days=14)
        else:
            try:
                date_start = datetime.strptime(date_start, '%Y-%m-%d')
            except ValueError:
                date_start = now - timedelta(days=7)

        if date_end in ['today', 'now']:
            date_end = now
        else:
            try:
                date_end = datetime.strptime(date_end, '%Y-%m-%d')
            except ValueError:
                date_end = now

        return date_start, date_end
