"""Interactive flood map generation."""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from ..utils.config import settings


class FloodMapper:
    """Generate interactive flood maps."""

    def __init__(self):
        """Initialize flood mapper."""
        self.output_dir = settings.output_dir / "maps"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_flood_map(
        self,
        flood_mask: np.ndarray,
        bbox: tuple,
        job_id: str,
        base_layer: str = "OpenStreetMap",
        overlay_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create interactive flood map.

        Args:
            flood_mask: Binary flood extent mask
            bbox: (min_lon, min_lat, max_lon, max_lat)
            job_id: Unique job identifier
            base_layer: Base map type
            overlay_data: Additional data to overlay

        Returns: Map file info
        """
        if not FOLIUM_AVAILABLE:
            return {'error': 'folium not available'}

        try:
            # Calculate center
            min_lon, min_lat, max_lon, max_lat = bbox
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles=base_layer
            )

            # Add flood extent layer
            self._add_flood_overlay(m, flood_mask, bbox)

            # Add additional overlays
            if overlay_data:
                self._add_overlays(m, overlay_data, bbox)

            # Add layer control
            folium.LayerControl().add_to(m)

            # Add fullscreen button
            plugins.Fullscreen().add_to(m)

            # Save map
            map_path = self.output_dir / f"flood_map_{job_id}.html"
            m.save(str(map_path))

            # Also save as PNG (requires selenium)
            png_path = None
            try:
                img_path = self.output_dir / f"flood_map_{job_id}.png"
                m.save(str(img_path))
                png_path = str(img_path)
            except Exception:
                pass

            return {
                'job_id': job_id,
                'map_path': str(map_path),
                'png_path': png_path,
                'bbox': bbox,
                'center': [center_lat, center_lon]
            }

        except Exception as e:
            return {'error': str(e)}

    def _add_flood_overlay(
        self,
        map_obj: folium.Map,
        flood_mask: np.ndarray,
        bbox: tuple
    ):
        """Add flood extent as polygon overlay."""
        min_lon, min_lat, max_lon, max_lat = bbox

        # Simple approach: draw rectangle for now
        # In production, would vectorize the actual mask

        # Create flood polygon (simplified bounding box for MVP)
        flood_polygon = [
            [min_lat, min_lon],
            [min_lat, max_lon],
            [max_lat, max_lon],
            [max_lat, min_lon],
            [min_lat, min_lon]
        ]

        # Calculate flood percentage for opacity
        flood_pct = np.mean(flood_mask) * 100 if flood_mask is not None else 50
        opacity = min(0.3 + (flood_pct / 200), 0.7)

        folium.Polygon(
            locations=flood_polygon,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=opacity,
            weight=2,
            popup=f'Flood Area: {flood_pct:.1f}% of view',
            tooltip='Flood Extent'
        ).add_to(map_obj)

        # Add border rectangle
        folium.Polygon(
            locations=[
                [min_lat, min_lon],
                [min_lat, max_lon],
                [max_lat, max_lon],
                [max_lat, min_lon],
                [min_lat, min_lon]
            ],
            color='red',
            fill=False,
            weight=2,
            dash_array='5, 5',
            tooltip='Analysis Area'
        ).add_to(map_obj)

    def _add_overlays(
        self,
        map_obj: folium.Map,
        overlay_data: Dict,
        bbox: tuple
    ):
        """Add additional data overlays."""
        # Add rainfall markers
        if 'rainfall_mm' in overlay_data:
            rainfall = overlay_data['rainfall_mm']
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2

            # Color based on intensity
            if rainfall > 100:
                color = 'darkred'
            elif rainfall > 50:
                color = 'orange'
            else:
                color = 'lightblue'

            folium.Marker(
                location=[center_lat, center_lon],
                popup=f'Rainfall: {rainfall:.1f} mm',
                icon=folium.Icon(color=color, icon='cloud', prefix='fa')
            ).add_to(map_obj)

        # Add affected infrastructure markers
        if 'affected_buildings' in overlay_data:
            buildings = overlay_data['affected_buildings']
            if buildings > 0:
                folium.Marker(
                    location=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2 + 0.1],
                    popup=f'~{buildings} buildings affected',
                    icon=folium.Icon(color='red', icon='home', prefix='fa')
                ).add_to(map_obj)

    def create_comparison_map(
        self,
        before_mask: np.ndarray,
        after_mask: np.ndarray,
        bbox: tuple,
        job_id: str
    ) -> str:
        """Create before/after comparison map."""
        if not FOLIUM_AVAILABLE:
            return None

        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # Create side-by-side map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Before layer
        self._add_flood_overlay(m, before_mask, bbox)

        # After layer (would use plugins.Draw or similar for actual comparison)
        # For MVP, just show combined

        map_path = self.output_dir / f"comparison_{job_id}.html"
        m.save(str(map_path))

        return str(map_path)

    def add_legend(
        self,
        map_obj: folium.Map,
        legend_items: List[Tuple[str, str]]
    ):
        """Add legend to map."""
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 150px; height: auto;
                    background-color: white; border: 2px solid grey;
                    z-index: 9999; font-size: 12px; padding: 10px">
        <p><b>Legend</b></p>
        '''

        for label, color in legend_items:
            legend_html += f'<p><i style="background:{color}; width:15px; height:15px; display:inline-block"></i> {label}</p>'

        legend_html += '</div>'
        map_obj.get_root().html.add_child(folium.Element(legend_html))
