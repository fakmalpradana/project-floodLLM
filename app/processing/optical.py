"""Optical imagery processing for flood validation."""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from ..utils.config import settings


class OpticalProcessor:
    """Process Sentinel-2 optical data for flood validation."""

    def __init__(self):
        """Initialize optical processor."""
        self.output_dir = settings.output_dir / "optical_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_ndwi(
        self,
        filepath: str,
        bbox: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate Normalized Difference Water Index (NDWI).

        NDWI = (Green - NIR) / (Green + NIR)
        Values > 0.3 typically indicate water

        Args:
            filepath: Path to Sentinel-2 GeoTIFF
            bbox: Optional area of interest

        Returns: NDWI analysis results
        """
        if not RASTERIO_AVAILABLE:
            print("rasterio not available")
            return None

        if not Path(filepath).exists():
            return None

        try:
            # Read required bands
            green, nir, swir, transform, crs = self._read_sentinel2_bands(filepath)

            if green is None:
                return None

            # Calculate NDWI
            ndwi = self._compute_ndwi(green, nir)

            # Calculate MNDWI (Modified NDWI for built-up areas)
            mndwi = self._compute_mndwi(green, swir) if swir is not None else None

            # Create water mask
            water_mask = ndwi > 0.3

            # Calculate statistics
            stats = self._calculate_water_stats(water_mask, transform)

            # Save results
            result = self._save_ndwi_results(
                ndwi, water_mask, transform, crs, filepath, stats
            )

            return result

        except Exception as e:
            print(f"Optical processing error: {e}")
            return None

    def _read_sentinel2_bands(
        self,
        filepath: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Any, Any]:
        """Read Sentinel-2 bands for water index calculation."""
        try:
            with rasterio.open(filepath) as src:
                # Sentinel-2 band order varies by processing
                # Typical order: B2, B3, B4, B8, B11 (Blue, Green, Red, NIR, SWIR)

                if src.count >= 5:
                    blue = src.read(1).astype(np.float32)
                    green = src.read(2).astype(np.float32)
                    red = src.read(3).astype(np.float32)
                    nir = src.read(4).astype(np.float32)
                    swir = src.read(5).astype(np.float32)
                else:
                    # Try to identify bands from metadata
                    print(f"Unexpected band count: {src.count}")
                    return None, None, None, None, None

                transform = src.transform
                crs = src.crs

                # Scale reflectance values (Sentinel-2 L2A has 0.0001 scale)
                # Check if values are already scaled
                if np.nanmean(green) > 1:
                    green = green / 10000
                    nir = nir / 10000
                    swir = swir / 10000

                # Mask invalid values
                green = np.ma.masked_invalid(green).filled(0)
                nir = np.ma.masked_invalid(nir).filled(0)
                swir = np.ma.masked_invalid(swir).filled(0)

                return green, nir, swir, transform, crs

        except Exception as e:
            print(f"Error reading Sentinel-2 file: {e}")
            return None, None, None, None, None

    def _compute_ndwi(
        self,
        green: np.ndarray,
        nir: np.ndarray
    ) -> np.ndarray:
        """Compute NDWI."""
        # Avoid division by zero
        denominator = green + nir
        denominator[denominator == 0] = 1e-10

        ndwi = (green - nir) / denominator

        # Clip to valid range
        ndwi = np.clip(ndwi, -1, 1)

        return ndwi

    def _compute_mndwi(
        self,
        green: np.ndarray,
        swir: np.ndarray
    ) -> np.ndarray:
        """Compute Modified NDWI (better for urban areas)."""
        denominator = green + swir
        denominator[denominator == 0] = 1e-10

        mndwi = (green - swir) / denominator

        return np.clip(mndwi, -1, 1)

    def _calculate_water_stats(
        self,
        mask: np.ndarray,
        transform: Any
    ) -> Dict[str, Any]:
        """Calculate water body statistics."""
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        water_pixels = np.sum(mask)
        total_pixels = mask.size

        water_area_m2 = water_pixels * pixel_area_m2
        water_area_km2 = water_area_m2 / 1_000_000

        water_percentage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        return {
            'water_pixels': int(water_pixels),
            'total_pixels': int(total_pixels),
            'water_area_km2': round(water_area_km2, 2),
            'water_percentage': round(water_percentage, 2)
        }

    def _save_ndwi_results(
        self,
        ndwi: np.ndarray,
        mask: np.ndarray,
        transform: Any,
        crs: Any,
        source_filepath: str,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save NDWI analysis results."""
        import uuid
        import json

        job_id = str(uuid.uuid4())[:8]

        # Save NDWI raster
        ndwi_path = self.output_dir / f"ndwi_{job_id}.tiff"

        try:
            with rasterio.open(
                ndwi_path,
                'w',
                driver='GTiff',
                height=ndwi.shape[0],
                width=ndwi.shape[1],
                count=1,
                dtype=np.float32,
                transform=transform,
                crs=crs
            ) as dst:
                dst.write(ndwi, 1)
        except Exception as e:
            print(f"Error saving NDWI: {e}")
            ndwi_path = None

        # Save water mask
        mask_path = self.output_dir / f"water_mask_{job_id}.tiff"

        try:
            with rasterio.open(
                mask_path,
                'w',
                driver='GTiff',
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=np.uint8,
                transform=transform,
                crs=crs
            ) as dst:
                dst.write(mask.astype(np.uint8), 1)
        except Exception as e:
            mask_path = None

        metadata = {
            'job_id': job_id,
            'source_file': source_filepath,
            'ndwi_path': str(ndwi_path),
            'mask_path': str(mask_path),
            'statistics': stats,
            'method': 'NDWI (threshold > 0.3)'
        }

        metadata_path = self.output_dir / f"ndwi_{job_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        metadata['metadata_path'] = str(metadata_path)
        return metadata

    def validate_flood_detection(
        self,
        sar_flood_mask: np.ndarray,
        optical_filepath: str
    ) -> Dict[str, Any]:
        """
        Validate SAR-based flood detection with optical data.

        Compares SAR water mask with NDWI-derived water mask.
        """
        ndwi_result = self.calculate_ndwi(optical_filepath)

        if ndwi_result is None:
            return {'valid': False, 'error': 'Could not process optical data'}

        # Load NDWI mask
        mask_path = ndwi_result.get('mask_path')
        if not mask_path or not Path(mask_path).exists():
            return {'valid': False, 'error': 'NDWI mask not found'}

        try:
            with rasterio.open(mask_path) as src:
                ndwi_mask = src.read(1).astype(bool)

            # Calculate agreement
            if sar_flood_mask.shape != ndwi_mask.shape:
                return {
                    'valid': False,
                    'error': 'Mask shape mismatch',
                    'note': 'Masks must have same resolution and extent'
                }

            # Intersection over Union
            intersection = np.sum(sar_flood_mask & ndwi_mask)
            union = np.sum(sar_flood_mask | ndwi_mask)

            iou = intersection / union if union > 0 else 0

            # Agreement percentage
            agreement = np.mean(sar_flood_mask == ndwi_mask) * 100

            return {
                'valid': True,
                'iou': round(iou, 3),
                'agreement_percentage': round(agreement, 1),
                'sar_flood_area_km2': np.sum(sar_flood_mask) * 0.0001,  # Approximate
                'ndwi_water_area_km2': ndwi_result['statistics']['water_area_km2'],
                'confidence': 'high' if iou > 0.7 else 'medium' if iou > 0.4 else 'low'
            }

        except Exception as e:
            return {'valid': False, 'error': str(e)}
