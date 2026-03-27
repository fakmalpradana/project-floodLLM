"""Sentinel-1 SAR processing for flood detection."""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from ..utils.config import settings


class SARProcessor:
    """Process Sentinel-1 SAR data for flood detection."""

    def __init__(self):
        """Initialize SAR processor."""
        self.output_dir = settings.output_dir / "flood_masks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.water_threshold = settings.water_threshold_vv

    def process(
        self,
        filepath: str,
        bbox: tuple,
        method: str = "otsu"
    ) -> Optional[Dict[str, Any]]:
        """
        Process Sentinel-1 image for flood detection.

        Args:
            filepath: Path to GeoTIFF file
            bbox: Area of interest (min_lon, min_lat, max_lon, max_lat)
            method: "otsu" for automatic threshold, "fixed" for manual

        Returns: Flood detection results
        """
        if not RASTERIO_AVAILABLE:
            print("rasterio not available. Cannot process SAR data.")
            return None

        if not Path(filepath).exists():
            print(f"File not found: {filepath}")
            return None

        try:
            # Read and preprocess
            vv_data, vh_data, transform, crs = self._read_sentinel1(filepath)

            if vv_data is None:
                return None

            # Apply threshold
            if method == "otsu" and SKIMAGE_AVAILABLE:
                threshold = self._calculate_otsu_threshold(vv_data)
            else:
                threshold = self.water_threshold

            # Create water mask
            water_mask = vv_data < threshold

            # Post-process mask
            water_mask = self._post_process_mask(water_mask)

            # Calculate flood statistics
            flood_stats = self._calculate_flood_stats(
                water_mask, transform, bbox
            )

            # Save results
            result = self._save_results(
                water_mask, transform, crs, filepath, flood_stats
            )

            return result

        except Exception as e:
            print(f"SAR processing error: {e}")
            return None

    def _read_sentinel1(
        self,
        filepath: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any, Any]:
        """Read and calibrate Sentinel-1 data."""
        try:
            with rasterio.open(filepath) as src:
                # Read VV and VH bands
                vv_data = src.read(1).astype(np.float32)
                vh_data = src.read(2).astype(np.float32) if src.count >= 2 else None

                transform = src.transform
                crs = src.crs

                # Convert to dB if needed (values > 0 suggest linear scale)
                if np.nanmean(vv_data[vv_data > 0]) > 1:
                    vv_data = 10 * np.log10(vv_data)
                if vh_data is not None and np.nanmean(vh_data[vh_data > 0]) > 1:
                    vh_data = 10 * np.log10(vh_data)

                # Mask invalid values
                vv_data = np.ma.masked_invalid(vv_data)

                return vv_data.filled(-9999), vh_data.filled(-9999) if vh_data is not None else None, transform, crs

        except Exception as e:
            print(f"Error reading Sentinel-1 file: {e}")
            return None, None, None, None

    def _calculate_otsu_threshold(self, data: np.ndarray) -> float:
        """Calculate optimal threshold using Otsu's method."""
        # Filter to reasonable backscatter values
        valid_data = data[(data > -30) & (data < 0)]

        if len(valid_data) < 100:
            return self.water_threshold

        try:
            threshold = threshold_otsu(valid_data)
            # Otsu finds the bimodal threshold - water is typically lower
            return min(threshold, self.water_threshold)
        except Exception:
            return self.water_threshold

    def _post_process_mask(
        self,
        mask: np.ndarray,
        min_size: int = 100
    ) -> np.ndarray:
        """Clean up water mask with morphological operations."""
        if not SKIMAGE_AVAILABLE:
            return mask

        # Remove small objects
        labeled, _ = self._label_connected(mask, min_size)

        # Fill holes
        filled = binary_fill_holes(labeled > 0)

        # Smooth boundaries
        smoothed = binary_opening(filled, structure=np.ones((3, 3)))
        smoothed = binary_closing(smoothed, structure=np.ones((3, 3)))

        # Remove border artifacts
        cleaned = clear_border(smoothed)

        return cleaned.astype(bool)

    def _label_connected(
        self,
        mask: np.ndarray,
        min_size: int = 0
    ) -> Tuple[np.ndarray, int]:
        """Label connected components."""
        try:
            from scipy.ndimage import label
            labeled, num_features = label(mask)

            if min_size > 0:
                # Remove small components
                component_sizes = np.bincount(labeled.ravel())
                small_components = np.where(component_sizes < min_size)[0]
                for comp in small_components:
                    labeled[labeled == comp] = 0

            return labeled, num_features

        except ImportError:
            return mask.astype(int), 0

    def _calculate_flood_stats(
        self,
        mask: np.ndarray,
        transform: Any,
        bbox: tuple
    ) -> Dict[str, Any]:
        """Calculate flood statistics."""
        # Pixel area from transform
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        # Count flooded pixels
        flooded_pixels = np.sum(mask)
        total_pixels = mask.size

        # Calculate area
        flood_area_m2 = flooded_pixels * pixel_area_m2
        flood_area_km2 = flood_area_m2 / 1_000_000

        # Calculate percentage
        flood_percentage = (flooded_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        return {
            'flooded_pixels': int(flooded_pixels),
            'total_pixels': int(total_pixels),
            'flood_area_km2': round(flood_area_km2, 2),
            'flood_percentage': round(flood_percentage, 2),
            'pixel_resolution_m': round(np.sqrt(pixel_area_m2), 1)
        }

    def _save_results(
        self,
        mask: np.ndarray,
        transform: Any,
        crs: Any,
        source_filepath: str,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save flood mask and metadata."""
        import uuid

        # Generate unique ID
        job_id = str(uuid.uuid4())[:8]

        # Save mask as GeoTIFF
        mask_path = self.output_dir / f"flood_mask_{job_id}.tiff"

        try:
            with rasterio.open(
                mask_path,
                'w',
                driver='GTiff',
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=mask.dtype,
                transform=transform,
                crs=crs
            ) as dst:
                dst.write(mask.astype(np.uint8), 1)

        except Exception as e:
            print(f"Error saving mask: {e}")
            mask_path = None

        # Save metadata
        metadata = {
            'job_id': job_id,
            'source_file': source_filepath,
            'mask_path': str(mask_path) if mask_path else None,
            'statistics': stats,
            'processing_params': {
                'threshold': self.water_threshold,
                'method': 'otsu'
            }
        }

        metadata_path = self.output_dir / f"flood_mask_{job_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        metadata['metadata_path'] = str(metadata_path)
        return metadata

    def create_geojson(
        self,
        mask: np.ndarray,
        transform: Any,
        crs: Any
    ) -> Dict:
        """Convert flood mask to GeoJSON."""
        from shapely.geometry import mapping, shape
        import geopandas as gpd

        # This would require additional processing to vectorize the raster
        # For now, return a placeholder
        return {
            "type": "FeatureCollection",
            "features": [],
            "note": "Vectorization requires additional processing"
        }
