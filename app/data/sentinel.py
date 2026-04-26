"""Sentinel-1 and Sentinel-2 data download."""
import os
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import ee
    from ee import batch
    EARTHENGINE_AVAILABLE = True
except ImportError:
    EARTHENGINE_AVAILABLE = False

from ..utils.config import settings


class SentinelDownloader:
    """Download Sentinel-1 SAR and Sentinel-2 optical data."""

    def __init__(self):
        """Initialize Sentinel downloader."""
        self.data_dir = settings.data_dir / "sentinel"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if EARTHENGINE_AVAILABLE:
            try:
                # Try to initialize with Service Account if credentials provided
                if settings.google_application_credentials and os.path.exists(settings.google_application_credentials):
                    print(f"Initializing Earth Engine with Service Account: {settings.gcp_service_account_email}")
                    credentials = ee.ServiceAccountCredentials(
                        settings.gcp_service_account_email,
                        settings.google_application_credentials
                    )
                    ee.Initialize(credentials)
                else:
                    # Fallback to default application credentials (standard for Cloud Run/App Engine)
                    print("Initializing Earth Engine with default credentials...")
                    ee.Initialize()
                
                self.ee_initialized = True
            except Exception as e:
                err = str(e)
                if "not registered" in err.lower() or "project" in err.lower():
                    print(
                        f"[GEE] Earth Engine init failed — project not registered. "
                        f"Register at https://code.earthengine.google.com/app/projects : {err}"
                    )
                else:
                    print(f"[GEE] Earth Engine init failed: {err}")
                self.ee_initialized = False
        else:
            self.ee_initialized = False

    async def download_sentinel1(
        self,
        bbox: tuple,
        date_start: str,
        date_end: str,
        max_images: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Download Sentinel-1 GRD products for flood detection.

        Priority: Google Earth Engine → Copernicus CDSE → empty list (simulation fallback).
        Returns empty list when no real data is available; the pipeline then uses
        simulated SAR data via VectorGenerator.
        """
        date_start_dt, date_end_dt = self._parse_dates(date_start, date_end)

        if self.ee_initialized:
            result = await self._download_sentinel1_ee(bbox, date_start_dt, date_end_dt, max_images)
            if result:
                return result
            print("[S1] GEE returned no images — trying Copernicus fallback")

        if settings.copernicus_username and settings.copernicus_password:
            print("[S1] Trying Copernicus CDSE download...")
            result = await self._download_sentinel1_copernicus(bbox, date_start_dt, date_end_dt, max_images)
            if result:
                return result
            print("[S1] Copernicus download failed — entering simulation mode")
        else:
            print("[S1] No satellite API available (GEE not registered, no Copernicus creds) — simulation mode")

        return []

    async def _download_sentinel1_ee(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Download Sentinel-1 using Earth Engine."""
        downloaded = []

        try:
            # Define area of interest
            aoi = ee.Geometry.Rectangle(bbox)

            # Filter Sentinel-1 GRD collection
            s1_collection = (
                ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(aoi)
                .filterDate(date_start, date_end)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .select(['VV', 'VH'])
            )

            # Get image list
            image_list = s1_collection.limit(max_images).getInfo()

            if not image_list.get('features'):
                print("No Sentinel-1 images found for area/date")
                return []

            # Download each image
            for feature in image_list['features']:
                props = feature['properties']
                image_id = props['system:id']

                # Create download task
                url = s1_collection.filter(ee.Filter.eq('system:id', image_id)) \
                    .first().getDownloadURL({
                        'name': f's1_{image_id.replace("/", "_")}',
                        'scale': 10,
                        'region': bbox
                    })

                # Download file
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=60.0)
                    if response.status_code == 200:
                        filepath = self.data_dir / f"{image_id.replace('/', '_')}.tiff"
                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                        downloaded.append({
                            'id': image_id,
                            'filepath': str(filepath),
                            'date': props['system:time_start'],
                            'bbox': bbox
                        })
                        print(f"Downloaded: {image_id}")

        except Exception as e:
            print(f"Sentinel-1 Earth Engine download error: {e}")

        return downloaded

    async def _download_sentinel1_copernicus(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Download Sentinel-1 using Copernicus Data Space API."""
        downloaded = []

        try:
            from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
            from shapely.geometry import box

            # CDSE (Copernicus Data Space Ecosystem) API
            api = SentinelAPI(
                settings.copernicus_username,
                settings.copernicus_password,
                'https://catalogue.dataspace.copernicus.eu/odata/v1'
            )

            # Define search area
            wkt = f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"

            # Search for products
            products = api.query(
                wkt,
                producttype='GRD',
                date=(date_start.strftime('%Y%m%d'), date_end.strftime('%Y%m%d')),
                limit=max_images
            )

            # Download products
            for uuid, props in products.items():
                filepath = api.download(uuid, self.data_dir)
                downloaded.append({
                    'id': props['identifier'],
                    'filepath': str(filepath),
                    'date': props['beginposition'],
                    'bbox': bbox
                })

        except ImportError:
            print("[S1] sentinelsat not installed — cannot use Copernicus API")
        except Exception as e:
            err = str(e)
            if "403" in err or "Forbidden" in err:
                print(
                    "[S1] Copernicus CDSE 403 Forbidden — action required: "
                    "log in at https://dataspace.copernicus.eu/ and accept the new Terms & Conditions"
                )
            elif "401" in err or "Unauthorized" in err:
                print("[S1] Copernicus CDSE 401 Unauthorized — check username/password in .env")
            else:
                print(f"[S1] Copernicus API error: {err}")

        return downloaded

    async def download_sentinel2(
        self,
        bbox: tuple,
        date_start: str,
        date_end: str,
        max_cloud_cover: float = 20,
        max_images: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Download Sentinel-2 L2A products for validation.

        Args:
            bbox: Bounding box
            date_start: Start date
            date_end: End date
            max_cloud_cover: Maximum cloud cover percentage
            max_images: Maximum images

        Returns: List of downloaded file info
        """
        date_start, date_end = self._parse_dates(date_start, date_end)

        if not self.ee_initialized:
            print("[S2] Earth Engine not available — Sentinel-2 skipped (optical validation will use simulation)")
            return []

        downloaded = []

        try:
            aoi = ee.Geometry.Rectangle(bbox)

            # Filter Sentinel-2 collection (Level-2A, surface reflectance)
            s2_collection = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(aoi)
                .filterDate(date_start, date_end)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                .select(['B2', 'B3', 'B4', 'B8', 'B11'])  # Required bands for NDWI
            )

            image_list = s2_collection.limit(max_images).getInfo()

            if not image_list.get('features'):
                print("No Sentinel-2 images found (may be cloudy)")
                return []

            for feature in image_list['features']:
                props = feature['properties']
                image_id = props['system:id']

                image = s2_collection.filter(ee.Filter.eq('system:id', image_id)).first()

                # Download as GeoTIFF
                url = image.getDownloadURL({
                    'name': f's2_{image_id.replace("/", "_")}',
                    'scale': 10,
                    'region': bbox,
                    'crs': 'EPSG:4326'
                })

                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=60.0)
                    if response.status_code == 200:
                        filepath = self.data_dir / f"{image_id.replace('/', '_')}.tiff"
                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                        downloaded.append({
                            'id': image_id,
                            'filepath': str(filepath),
                            'date': props['system:time_start'],
                            'cloud_cover': props.get('CLOUDY_PIXEL_PERCENTAGE', 0),
                            'bbox': bbox
                        })

        except Exception as e:
            print(f"Sentinel-2 download error: {e}")

        return downloaded

    def _parse_dates(self, date_start: str, date_end: str) -> tuple:
        """Parse date strings to datetime objects."""
        now = datetime.utcnow()

        # Handle relative dates
        regex_dt = None
        if isinstance(date_start, str):
            # Regex patterns for English and Indonesian
            # Years
            m = re.search(r'(\d+)\s*tahun', date_start, re.IGNORECASE) or \
                re.search(r'last\s*(\d+)\s*year', date_start, re.IGNORECASE) or \
                re.search(r'(\d+)\s*years?\s*ago', date_start, re.IGNORECASE)
            if m:
                regex_dt = now - timedelta(days=int(m.group(1)) * 365)
            
            # Months
            if not regex_dt:
                m = re.search(r'(\d+)\s*bulan', date_start, re.IGNORECASE) or \
                    re.search(r'last\s*(\d+)\s*month', date_start, re.IGNORECASE) or \
                    re.search(r'(\d+)\s*months?\s*ago', date_start, re.IGNORECASE)
                if m:
                    regex_dt = now - timedelta(days=int(m.group(1)) * 30)
            
            # Weeks
            if not regex_dt:
                m = re.search(r'(\d+)\s*minggu', date_start, re.IGNORECASE) or \
                    re.search(r'last\s*(\d+)\s*week', date_start, re.IGNORECASE) or \
                    re.search(r'(\d+)\s*weeks?\s*ago', date_start, re.IGNORECASE)
                if m:
                    regex_dt = now - timedelta(days=int(m.group(1)) * 7)
            
            # Days
            if not regex_dt:
                m = re.search(r'(\d+)\s*hari', date_start, re.IGNORECASE) or \
                    re.search(r'last\s*(\d+)\s*day', date_start, re.IGNORECASE) or \
                    re.search(r'(\d+)\s*days?\s*ago', date_start, re.IGNORECASE)
                if m:
                    regex_dt = now - timedelta(days=int(m.group(1)))
            
            # Fixed patterns
            if not regex_dt:
                if re.search(r'tahun\s*lalu', date_start, re.IGNORECASE):
                    regex_dt = now - timedelta(days=365)
                elif re.search(r'bulan\s*lalu', date_start, re.IGNORECASE):
                    regex_dt = now - timedelta(days=30)
                elif re.search(r'minggu\s*lalu', date_start, re.IGNORECASE):
                    regex_dt = now - timedelta(days=7)
                elif re.search(r'kemarin', date_start, re.IGNORECASE):
                    regex_dt = now - timedelta(days=1)

        if regex_dt:
            date_start = regex_dt
        elif date_start in ['last 7 days', 'past week', '1 week']:
            date_start = now - timedelta(days=7)
        elif date_start in ['last 14 days', 'past 2 weeks']:
            date_start = now - timedelta(days=14)
        elif date_start in ['last 30 days', 'past month']:
            date_start = now - timedelta(days=30)
        else:
            try:
                if isinstance(date_start, str):
                    date_start = datetime.strptime(date_start, '%Y-%m-%d')
            except (ValueError, TypeError):
                date_start = now - timedelta(days=7)

        # Handle end date
        if date_end in ['today', 'now']:
            date_end = now
        else:
            try:
                if isinstance(date_end, str):
                    date_end = datetime.strptime(date_end, '%Y-%m-%d')
            except (ValueError, TypeError):
                date_end = now

        return date_start, date_end
