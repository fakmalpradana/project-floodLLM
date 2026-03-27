"""Sentinel-1 and Sentinel-2 data download."""
import os
import asyncio
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

        if EARTHENGINE_AVAILABLE and settings.google_api_key:
            try:
                ee.Initialize()
                self.ee_initialized = True
            except Exception as e:
                print(f"Earth Engine init failed: {e}")
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

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            date_start: Start date (YYYY-MM-DD or relative)
            date_end: End date
            max_images: Maximum number of images to download

        Returns: List of downloaded file info
        """
        # Parse dates
        date_start, date_end = self._parse_dates(date_start, date_end)

        if self.ee_initialized:
            return await self._download_sentinel1_ee(bbox, date_start, date_end, max_images)
        else:
            # Fallback: use Copernicus API
            return await self._download_sentinel1_copernicus(bbox, date_start, date_end, max_images)

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
            from sentinelsat import SentinelAPI, geojson_to_wkt
            from shapely.geometry import box

            api = SentinelAPI(
                settings.copernicus_username,
                settings.copernicus_password,
                'https://apihub.copernicus.eu/apihub'
            )

            # Define search area
            geometry = box(*bbox)
            wkt = geojson_to_wkt(geometry)

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
            print("sentinelsat not installed. Using Earth Engine fallback.")
        except Exception as e:
            print(f"Copernicus API download error: {e}")

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
            print("Earth Engine not available. Sentinel-2 download skipped.")
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
        if date_start in ['last 7 days', 'past week', '1 week']:
            date_start = now - timedelta(days=7)
        elif date_start in ['last 14 days', 'past 2 weeks']:
            date_start = now - timedelta(days=14)
        elif date_start in ['last 30 days', 'past month']:
            date_start = now - timedelta(days=30)
        else:
            try:
                date_start = datetime.strptime(date_start, '%Y-%m-%d')
            except ValueError:
                date_start = now - timedelta(days=7)

        # Handle end date
        if date_end in ['today', 'now']:
            date_end = now
        else:
            try:
                date_end = datetime.strptime(date_end, '%Y-%m-%d')
            except ValueError:
                date_end = now

        return date_start, date_end
