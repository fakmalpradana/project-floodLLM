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
                if settings.google_application_credentials and os.path.exists(settings.google_application_credentials):
                    print(f"Initializing Earth Engine with Service Account: {settings.gcp_service_account_email}")
                    credentials = ee.ServiceAccountCredentials(
                        settings.gcp_service_account_email,
                        settings.google_application_credentials
                    )
                    ee.Initialize(credentials)
                else:
                    print("Initializing Earth Engine with default credentials...")
                    ee.Initialize()

                self.ee_initialized = True
            except Exception as e:
                err = str(e)
                if "not registered" in err.lower() or "project" in err.lower():
                    print(
                        f"[GEE] Earth Engine init failed — project not registered. "
                        f"Register at https://console.cloud.google.com/earth-engine : {err}"
                    )
                else:
                    print(f"[GEE] Earth Engine init failed: {err}")
                self.ee_initialized = False
        else:
            self.ee_initialized = False

    # ─────────────────────────────────────────────────────────────────────────
    # Sentinel-1
    # ─────────────────────────────────────────────────────────────────────────

    async def download_sentinel1(
        self,
        bbox: tuple,
        date_start: str,
        date_end: str,
        max_images: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Download Sentinel-1 GRD products for flood detection.

        Priority: GEE → Planetary Computer STAC → empty list (simulation fallback).
        """
        date_start_dt, date_end_dt = self._parse_dates(date_start, date_end)

        if self.ee_initialized:
            result = await self._download_sentinel1_ee(bbox, date_start_dt, date_end_dt, max_images)
            if result:
                return result
            print("[S1] GEE returned no images — trying Planetary Computer fallback")

        result = await self._download_sentinel1_planetary_computer(bbox, date_start_dt, date_end_dt, max_images)
        if result:
            return result

        print("[S1] All sources exhausted — entering simulation mode")
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
            aoi = ee.Geometry.Rectangle(bbox)

            s1_collection = (
                ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(aoi)
                .filterDate(date_start, date_end)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .select(['VV', 'VH'])
            )

            image_list = s1_collection.limit(max_images).getInfo()

            if not image_list.get('features'):
                print("No Sentinel-1 images found for area/date")
                return []

            for feature in image_list['features']:
                props = feature['properties']
                image_id = props['system:id']

                url = s1_collection.filter(ee.Filter.eq('system:id', image_id)) \
                    .first().getDownloadURL({
                        'name': f's1_{image_id.replace("/", "_")}',
                        'scale': 10,
                        'region': bbox
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
                            'bbox': bbox,
                            'source': 'gee'
                        })
                        print(f"[S1] Downloaded via GEE: {image_id}")

        except Exception as e:
            print(f"[S1] Earth Engine download error: {e}")

        return downloaded

    async def _download_sentinel1_planetary_computer(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Download Sentinel-1 RTC via Planetary Computer STAC (no auth). Runs in thread pool."""
        try:
            return await asyncio.to_thread(
                self._sync_dl_s1_pc, bbox, date_start, date_end, max_images
            )
        except ImportError:
            print("[S1] pystac-client or planetary-computer not installed — skipping Planetary Computer")
        except Exception as e:
            print(f"[S1] Planetary Computer error: {e}")
        return []

    def _sync_dl_s1_pc(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Synchronous worker: search Planetary Computer and read COG for bbox at 100m resolution."""
        import os
        os.environ.setdefault('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
        os.environ.setdefault('GDAL_HTTP_MULTIPLEX', 'YES')
        os.environ.setdefault('GDAL_HTTP_VERSION', '2')
        os.environ.setdefault('CPL_VSIL_CURL_CACHE_SIZE', '200000000')

        from pystac_client import Client
        import planetary_computer
        import rasterio
        from rasterio.warp import transform_bounds
        import rasterio.windows as rw
        import numpy as np

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=list(bbox),
            datetime=f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}",
            max_items=max_images,
        )
        items = list(search.items())
        if not items:
            print("[S1] Planetary Computer: no Sentinel-1 RTC images found")
            return []

        downloaded = []
        for item in items[:max_images]:
            if 'vv' not in item.assets:
                continue
            vv_url = item.assets['vv'].href
            try:
                with rasterio.open(vv_url) as src:
                    img_bounds = transform_bounds('EPSG:4326', src.crs, *bbox)
                    window = rw.from_bounds(*img_bounds, transform=src.transform)
                    col_off = max(0, int(window.col_off))
                    row_off = max(0, int(window.row_off))
                    col_end = min(src.width, int(window.col_off + window.width))
                    row_end = min(src.height, int(window.row_off + window.height))
                    if col_end <= col_off or row_end <= row_off:
                        continue
                    window_clamped = rw.Window(col_off, row_off, col_end - col_off, row_end - row_off)

                    # Downsample to ~100m resolution to limit HTTP range requests.
                    # At 10m native res, a 60km×65km bbox = 6000×6500px = 600+ HTTP requests.
                    # At 100m (GDAL overview) = 600×650px = 6 HTTP requests → ~100x faster.
                    target = 600
                    h_px = row_end - row_off
                    w_px = col_end - col_off
                    scale = max(1, max(h_px, w_px) / target)
                    out_h = max(100, int(h_px / scale))
                    out_w = max(100, int(w_px / scale))

                    data = src.read(
                        1, window=window_clamped,
                        out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.average,
                    ).astype(np.float32)
                    out_transform = src.window_transform(window_clamped)
                    data_db = 10 * np.log10(np.where(data > 0, data, 1e-10)).astype(np.float32)

                    filepath = self.data_dir / f"s1_rtc_{item.id}.tiff"
                    with rasterio.open(
                        filepath, 'w',
                        driver='GTiff', dtype='float32',
                        count=1, crs=src.crs,
                        transform=out_transform,
                        width=out_w,
                        height=out_h,
                    ) as dst:
                        dst.write(data_db, 1)

                downloaded.append({
                    'id': item.id,
                    'filepath': str(filepath),
                    'date': item.datetime.isoformat() if item.datetime else date_start.isoformat(),
                    'bbox': bbox,
                    'source': 'planetary_computer',
                })
                print(f"[S1] Downloaded from Planetary Computer: {item.id}")

            except Exception as e:
                print(f"[S1] Error downloading {item.id}: {e}")
                continue

        return downloaded

    # ─────────────────────────────────────────────────────────────────────────
    # Sentinel-2
    # ─────────────────────────────────────────────────────────────────────────

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

        Priority: GEE → Planetary Computer STAC → Element84 STAC → empty list.
        """
        date_start_dt, date_end_dt = self._parse_dates(date_start, date_end)

        if self.ee_initialized:
            result = await self._download_sentinel2_ee(bbox, date_start_dt, date_end_dt, max_cloud_cover, max_images)
            if result:
                return result
            print("[S2] GEE returned no images — trying STAC fallback")

        result = await self._download_sentinel2_stac(bbox, date_start_dt, date_end_dt, max_cloud_cover, max_images)
        if result:
            return result

        print("[S2] All sources exhausted — optical validation will use simulation")
        return []

    async def _download_sentinel2_ee(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime,
        max_cloud_cover: float,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Download Sentinel-2 L2A using Earth Engine."""
        downloaded = []

        try:
            aoi = ee.Geometry.Rectangle(bbox)

            s2_collection = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(aoi)
                .filterDate(date_start, date_end)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                .select(['B2', 'B3', 'B4', 'B8', 'B11'])
            )

            image_list = s2_collection.limit(max_images).getInfo()

            if not image_list.get('features'):
                print("[S2] No Sentinel-2 images found via GEE (may be cloudy)")
                return []

            for feature in image_list['features']:
                props = feature['properties']
                image_id = props['system:id']

                image = s2_collection.filter(ee.Filter.eq('system:id', image_id)).first()
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
                            'bbox': bbox,
                            'source': 'gee'
                        })

        except Exception as e:
            print(f"[S2] Earth Engine download error: {e}")

        return downloaded

    async def _download_sentinel2_stac(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime,
        max_cloud_cover: float,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """
        Download Sentinel-2 L2A via STAC (Planetary Computer → Element84).
        Runs blocking STAC search + rasterio in a thread pool.
        Saves 2-band GeoTIFF (band 1=green, band 2=NIR) for NDWI.
        """
        try:
            return await asyncio.to_thread(
                self._sync_dl_s2_stac, bbox, date_start, date_end, max_cloud_cover, max_images
            )
        except ImportError:
            print("[S2] pystac-client not installed — skipping STAC download")
        except Exception as e:
            print(f"[S2] STAC download error: {e}")
        return []

    def _sync_dl_s2_stac(
        self,
        bbox: tuple,
        date_start: datetime,
        date_end: datetime,
        max_cloud_cover: float,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Synchronous worker: search STAC and read COG bands for bbox."""
        downloaded = []

        # STAC endpoints to try in order
        stac_sources = [
            {
                "name": "Planetary Computer",
                "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
                "use_planetary_computer": True,
                "collection": "sentinel-2-l2a",
                "green_key": "B03",
                "nir_key": "B08",
                "cloud_prop": "eo:cloud_cover",
            },
            {
                "name": "Element84 Earth Search",
                "url": "https://earth-search.aws.element84.com/v1",
                "use_planetary_computer": False,
                "collection": "sentinel-2-l2a",
                "green_key": "green",
                "nir_key": "nir",
                "cloud_prop": "eo:cloud_cover",
            },
        ]

        try:
            from pystac_client import Client
            import rasterio
            from rasterio.warp import transform_bounds
            import rasterio.windows as rw
            import numpy as np

            for source in stac_sources:
                if downloaded:
                    break
                try:
                    kwargs = {}
                    if source["use_planetary_computer"]:
                        import planetary_computer
                        kwargs["modifier"] = planetary_computer.sign_inplace

                    catalog = Client.open(source["url"], **kwargs)
                    search = catalog.search(
                        collections=[source["collection"]],
                        bbox=list(bbox),
                        datetime=f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}",
                        query={source["cloud_prop"]: {"lt": max_cloud_cover}},
                        max_items=max_images,
                    )
                    items = list(search.items())
                    if not items:
                        print(f"[S2] {source['name']}: no images found")
                        continue

                    for item in items[:max_images]:
                        # Try configured keys then fallbacks
                        green_url = self._get_asset_url(item, [source["green_key"], "B03", "green"])
                        nir_url = self._get_asset_url(item, [source["nir_key"], "B08", "nir"])
                        if not green_url or not nir_url:
                            continue

                        try:
                            green_band = self._read_cog_window(green_url, bbox)
                            if green_band is None:
                                continue
                            data, out_transform, out_crs = green_band

                            nir_raw = self._read_cog_window(nir_url, bbox, out_shape=data.shape)
                            if nir_raw is None:
                                continue
                            nir_data, _, _ = nir_raw

                            filepath = self.data_dir / f"s2_ndwi_{item.id}.tiff"
                            with rasterio.open(
                                filepath, 'w',
                                driver='GTiff', dtype='float32',
                                count=2, crs=out_crs,
                                transform=out_transform,
                                width=data.shape[1], height=data.shape[0],
                            ) as dst:
                                dst.write(data, 1)
                                dst.write(nir_data, 2)

                            downloaded.append({
                                'id': item.id,
                                'filepath': str(filepath),
                                'date': item.datetime.isoformat() if item.datetime else date_start.isoformat(),
                                'cloud_cover': item.properties.get(source["cloud_prop"], 0),
                                'bbox': bbox,
                                'source': source["name"],
                            })
                            print(f"[S2] Downloaded from {source['name']}: {item.id}")

                        except Exception as e:
                            print(f"[S2] Error downloading item {item.id}: {e}")
                            continue

                except ImportError:
                    if source["use_planetary_computer"]:
                        print("[S2] planetary-computer package not installed — skipping Planetary Computer")
                    continue
                except Exception as e:
                    print(f"[S2] {source['name']} error: {e}")
                    continue

        except ImportError:
            print("[S2] pystac-client not installed — skipping STAC download")
        except Exception as e:
            print(f"[S2] STAC download error: {e}")

        return downloaded

    def _get_asset_url(self, item, keys: list) -> Optional[str]:
        """Return the first matching asset href from a list of candidate keys."""
        for key in keys:
            if key in item.assets:
                return item.assets[key].href
        return None

    def _read_cog_window(
        self,
        url: str,
        bbox: tuple,
        out_shape: Optional[tuple] = None
    ) -> Optional[tuple]:
        """
        Read a window of a COG matching bbox, downsampled to ≤600px on longest side.
        Returns (data_float32, transform, crs) or None on failure.
        """
        try:
            import os
            os.environ.setdefault('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
            os.environ.setdefault('GDAL_HTTP_MULTIPLEX', 'YES')
            os.environ.setdefault('GDAL_HTTP_VERSION', '2')

            import rasterio
            from rasterio.warp import transform_bounds
            import rasterio.windows as rw
            import numpy as np

            with rasterio.open(url) as src:
                img_bounds = transform_bounds('EPSG:4326', src.crs, *bbox)
                window = rw.from_bounds(*img_bounds, transform=src.transform)
                col_off = max(0, int(window.col_off))
                row_off = max(0, int(window.row_off))
                col_end = min(src.width, int(window.col_off + window.width))
                row_end = min(src.height, int(window.row_off + window.height))
                if col_end <= col_off or row_end <= row_off:
                    return None
                window_clamped = rw.Window(col_off, row_off, col_end - col_off, row_end - row_off)

                if out_shape is not None:
                    target_shape = out_shape
                else:
                    h_px = row_end - row_off
                    w_px = col_end - col_off
                    scale = max(1, max(h_px, w_px) / 600)
                    target_shape = (max(100, int(h_px / scale)), max(100, int(w_px / scale)))

                data = src.read(
                    1, window=window_clamped,
                    out_shape=target_shape,
                    resampling=rasterio.enums.Resampling.average,
                ).astype(np.float32)

                out_transform = src.window_transform(window_clamped)
                return data, out_transform, src.crs

        except Exception as e:
            print(f"[STAC] COG read error for {url}: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Date parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_dates(self, date_start: str, date_end: str) -> tuple:
        """Parse date strings to datetime objects."""
        now = datetime.utcnow()

        regex_dt = None
        if isinstance(date_start, str):
            m = re.search(r'(\d+)\s*tahun', date_start, re.IGNORECASE) or \
                re.search(r'last\s*(\d+)\s*year', date_start, re.IGNORECASE) or \
                re.search(r'(\d+)\s*years?\s*ago', date_start, re.IGNORECASE)
            if m:
                regex_dt = now - timedelta(days=int(m.group(1)) * 365)

            if not regex_dt:
                m = re.search(r'(\d+)\s*bulan', date_start, re.IGNORECASE) or \
                    re.search(r'last\s*(\d+)\s*month', date_start, re.IGNORECASE) or \
                    re.search(r'(\d+)\s*months?\s*ago', date_start, re.IGNORECASE)
                if m:
                    regex_dt = now - timedelta(days=int(m.group(1)) * 30)

            if not regex_dt:
                m = re.search(r'(\d+)\s*minggu', date_start, re.IGNORECASE) or \
                    re.search(r'last\s*(\d+)\s*week', date_start, re.IGNORECASE) or \
                    re.search(r'(\d+)\s*weeks?\s*ago', date_start, re.IGNORECASE)
                if m:
                    regex_dt = now - timedelta(days=int(m.group(1)) * 7)

            if not regex_dt:
                m = re.search(r'(\d+)\s*hari', date_start, re.IGNORECASE) or \
                    re.search(r'last\s*(\d+)\s*day', date_start, re.IGNORECASE) or \
                    re.search(r'(\d+)\s*days?\s*ago', date_start, re.IGNORECASE)
                if m:
                    regex_dt = now - timedelta(days=int(m.group(1)))

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

        if date_end in ['today', 'now']:
            date_end = now
        else:
            try:
                if isinstance(date_end, str):
                    date_end = datetime.strptime(date_end, '%Y-%m-%d')
            except (ValueError, TypeError):
                date_end = now

        return date_start, date_end
