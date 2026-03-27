"""Flood risk prediction model."""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

from ..utils.config import settings


class FloodRiskModel:
    """Simple flood risk prediction based on terrain and rainfall."""

    def __init__(self):
        """Initialize flood risk model."""
        self.output_dir = settings.output_dir / "risk_maps"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_risk(
        self,
        bbox: tuple,
        rainfall_mm: float,
        flood_extent: Optional[np.ndarray] = None,
        dem_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Predict flood risk based on multiple factors.

        Args:
            bbox: Area of interest (min_lon, min_lat, max_lon, max_lat)
            rainfall_mm: Total rainfall in mm
            flood_extent: Current flood mask (optional)
            dem_data: Digital elevation model data (optional)

        Returns: Risk assessment results
        """
        # Generate risk factors
        risk_factors = self._calculate_risk_factors(bbox, rainfall_mm, dem_data)

        # Combine factors into risk score
        risk_map, risk_stats = self._combine_risk_factors(
            risk_factors, flood_extent, rainfall_mm
        )

        # Generate risk zones
        risk_zones = self._classify_risk_zones(risk_map)

        result = {
            'bbox': bbox,
            'rainfall_mm': rainfall_mm,
            'risk_factors': risk_factors,
            'risk_statistics': risk_stats,
            'risk_zones': risk_zones,
            'recommendations': self._generate_recommendations(risk_zones, rainfall_mm)
        }

        # Save results
        self._save_risk_results(result)

        return result

    def _calculate_risk_factors(
        self,
        bbox: tuple,
        rainfall_mm: float,
        dem_data: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate individual risk factors."""
        min_lon, min_lat, max_lon, max_lat = bbox

        # Create grid for analysis
        lat_range = np.linspace(min_lat, max_lat, 100)
        lon_range = np.linspace(min_lon, max_lon, 100)
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

        factors = {}

        # Factor 1: Elevation risk (lower = higher risk)
        if dem_data is not None and dem_data.shape == (100, 100):
            elevation_normalized = (dem_data - dem_data.min()) / (dem_data.max() - dem_data.min() + 1e-10)
            factors['elevation'] = 1 - elevation_normalized  # Invert: lower = riskier
        else:
            # Simplified: assume areas near water bodies are lower
            distance_from_center = np.sqrt(
                (lon_grid - np.mean(lon_range)) ** 2 +
                (lat_grid - np.mean(lat_range)) ** 2
            )
            factors['elevation'] = 1 - (distance_from_center / distance_from_center.max())

        # Factor 2: Rainfall intensity risk
        # Higher rainfall = higher risk, saturates at ~200mm
        rainfall_factor = np.minimum(rainfall_mm / 200, 1.0)
        factors['rainfall'] = np.full((100, 100), rainfall_factor)

        # Factor 3: Slope risk (flatter = higher flood risk)
        # Simplified: edges of bbox assumed to have steeper terrain
        slope_factor = 1 - factors['elevation'] * 0.5  # Correlated with elevation
        factors['slope'] = slope_factor

        # Factor 4: Proximity to rivers (simplified)
        # Assume rivers flow through lower elevations
        factors['river_proximity'] = factors['elevation'] * 0.8

        return factors

    def _combine_risk_factors(
        self,
        factors: Dict[str, np.ndarray],
        flood_extent: Optional[np.ndarray],
        rainfall_mm: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Combine risk factors into composite risk map."""
        # Weighted combination
        weights = {
            'elevation': 0.3,
            'rainfall': 0.3,
            'slope': 0.2,
            'river_proximity': 0.2
        }

        risk_map = np.zeros(list(factors.values())[0].shape)

        for factor_name, factor_data in factors.items():
            risk_map += weights.get(factor_name, 0.25) * factor_data

        # Normalize to 0-1
        risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-10)

        # Adjust based on current flood extent (if available)
        if flood_extent is not None:
            # Areas currently flooded get highest risk
            risk_map = np.maximum(risk_map, flood_extent.astype(float) * 0.9)

        # Calculate statistics
        stats = {
            'mean_risk': float(np.mean(risk_map)),
            'max_risk': float(np.max(risk_map)),
            'high_risk_area_pct': float(np.mean(risk_map > 0.7) * 100),
            'moderate_risk_area_pct': float(np.mean((risk_map > 0.4) & (risk_map <= 0.7)) * 100),
            'low_risk_area_pct': float(np.mean(risk_map <= 0.4) * 100)
        }

        return risk_map, stats

    def _classify_risk_zones(
        self,
        risk_map: np.ndarray
    ) -> Dict[str, Any]:
        """Classify risk zones."""
        high_risk = risk_map > 0.7
        moderate_risk = (risk_map > 0.4) & (risk_map <= 0.7)
        low_risk = risk_map <= 0.4

        return {
            'high_risk_pixels': int(np.sum(high_risk)),
            'moderate_risk_pixels': int(np.sum(moderate_risk)),
            'low_risk_pixels': int(np.sum(low_risk)),
            'total_pixels': int(risk_map.size),
            'high_risk_map': high_risk.astype(np.uint8).tolist(),
            'moderate_risk_map': moderate_risk.astype(np.uint8).tolist(),
            'low_risk_map': low_risk.astype(np.uint8).tolist()
        }

    def _generate_recommendations(
        self,
        risk_zones: Dict[str, Any],
        rainfall_mm: float
    ) -> List[str]:
        """Generate recommendations based on risk assessment."""
        recommendations = []

        high_risk_pct = risk_zones['high_risk_pixels'] / risk_zones['total_pixels'] * 100

        if high_risk_pct > 30:
            recommendations.append("URGENT: Large area at high risk - consider immediate evacuation orders")
        elif high_risk_pct > 15:
            recommendations.append("WARNING: Significant high-risk area - prepare emergency response")
        else:
            recommendations.append("Monitor situation - current risk levels manageable")

        if rainfall_mm > 100:
            recommendations.append("Heavy rainfall detected - risk of flash flooding in low-lying areas")
        elif rainfall_mm > 50:
            recommendations.append("Moderate rainfall - continue monitoring water levels")

        recommendations.append("Coordinate with local emergency services")
        recommendations.append("Keep emergency supplies ready in high-risk zones")

        return recommendations

    def _save_risk_results(
        self,
        result: Dict[str, Any]
    ) -> str:
        """Save risk assessment results."""
        import uuid

        job_id = str(uuid.uuid4())[:8]
        metadata_path = self.output_dir / f"risk_{job_id}.json"

        # Don't save full risk maps in JSON (too large)
        save_result = {
            'job_id': job_id,
            'bbox': result['bbox'],
            'rainfall_mm': result['rainfall_mm'],
            'risk_statistics': result['risk_statistics'],
            'risk_zones_summary': {
                'high_risk_pixels': result['risk_zones']['high_risk_pixels'],
                'moderate_risk_pixels': result['risk_zones']['moderate_risk_pixels'],
                'low_risk_pixels': result['risk_zones']['low_risk_pixels']
            },
            'recommendations': result['recommendations']
        }

        with open(metadata_path, 'w') as f:
            json.dump(save_result, f, indent=2)

        return str(metadata_path)
