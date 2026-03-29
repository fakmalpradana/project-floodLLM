"""Enhanced interactive map with vector layers for flood risk & impact assessment.

Creates a publication-ready Folium map with:
- Layer 1: Flood extent polygons (Sentinel-1 + Sentinel-2 fusion)
- Layer 2: Flood risk zones (HIGH/MEDIUM/LOW)
- Layer 3: Impact buffer zones (0/500/1000/2000m)
- Layer 4: District statistics
- Statistics panel, legend, layer toggles
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import folium
    from folium import plugins, FeatureGroup, LayerControl, GeoJson, GeoJsonTooltip, GeoJsonPopup
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from ..utils.config import settings


# Jakarta bounding box for default view
JAKARTA_BOUNDS = [[-6.37, 106.65], [-6.05, 107.00]]


class VectorFloodMap:
    """Generate interactive flood map with multiple vector layers."""

    def __init__(self):
        self.output_dir = settings.output_dir / "maps"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_vector_map(
        self,
        job_id: str,
        bbox: tuple,
        flood_extent_geojson: Optional[Dict] = None,
        risk_zones_geojson: Optional[Dict] = None,
        impact_zones_geojson: Optional[Dict] = None,
        districts_geojson: Optional[Dict] = None,
        analysis_stats: Optional[Dict] = None,
        title: str = "Flood Risk & Impact Assessment",
        analysis_period: str = "January 2025"
    ) -> Dict[str, Any]:
        """
        Create interactive map with all vector layers.

        Returns: Dict with map path and metadata
        """
        if not FOLIUM_AVAILABLE:
            return {"error": "folium not available", "map_path": None}

        try:
            min_lon, min_lat, max_lon, max_lat = bbox
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles=None,  # we'll add tiles manually
                prefer_canvas=True
            )

            # Base tile layers (switchable)
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Satellite Imagery",
                overlay=False
            ).add_to(m)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Light Gray",
                overlay=False
            ).add_to(m)

            # Layer 4: Districts (bottom layer, always shown)
            if districts_geojson and districts_geojson.get("features"):
                self._add_district_layer(m, districts_geojson)

            # Layer 3: Impact zones (underneath risk + flood)
            if impact_zones_geojson and impact_zones_geojson.get("features"):
                self._add_impact_zones_layer(m, impact_zones_geojson)

            # Layer 2: Risk zones
            if risk_zones_geojson and risk_zones_geojson.get("features"):
                self._add_risk_zones_layer(m, risk_zones_geojson)

            # Layer 1: Flood extent (top layer)
            if flood_extent_geojson and flood_extent_geojson.get("features"):
                self._add_flood_extent_layer(m, flood_extent_geojson)

            # Analysis area boundary
            self._add_analysis_boundary(m, bbox)

            # Layer control
            LayerControl(collapsed=False).add_to(m)

            # Plugins
            plugins.Fullscreen(position="topright").add_to(m)
            plugins.MeasureControl(position="bottomleft").add_to(m)

            # Legend
            self._add_legend(m, title, analysis_period)

            # Statistics panel
            if analysis_stats:
                self._add_stats_panel(m, analysis_stats, analysis_period)

            # Title panel
            self._add_title_panel(m, title, analysis_period)

            # Save
            map_path = self.output_dir / f"flood_map_{job_id}.html"
            m.save(str(map_path))

            return {
                "job_id": job_id,
                "map_path": str(map_path),
                "bbox": bbox,
                "center": [center_lat, center_lon],
                "layers": {
                    "flood_extent": flood_extent_geojson is not None,
                    "risk_zones": risk_zones_geojson is not None,
                    "impact_zones": impact_zones_geojson is not None,
                    "districts": districts_geojson is not None
                }
            }

        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc(), "map_path": None}

    # ── Layer builders ────────────────────────────────────────────────────────

    def _add_flood_extent_layer(self, m: folium.Map, geojson: Dict):
        """Add flood extent polygon layer with styling and popups."""
        fg = FeatureGroup(name="Flood Extent (Sentinel-1+2)", show=True)

        def style_function(feature):
            conf = feature["properties"].get("confidence", "MEDIUM")
            if conf == "HIGH":
                fill_color = "#001f7a"
                color = "#00008B"
            else:
                fill_color = "#0044aa"
                color = "#0000CD"
            return {
                "fillColor": fill_color,
                "color": color,
                "weight": 2,
                "fillOpacity": 0.65,
                "opacity": 0.9
            }

        def highlight_function(feature):
            return {"weight": 3, "color": "#00FFFF", "fillOpacity": 0.75}

        popup = GeoJsonPopup(
            fields=["flood_type", "confidence", "area_km2", "area_ha", "source", "date_detected"],
            aliases=["Flood Type", "Confidence", "Area (km²)", "Area (ha)", "Data Source", "Date Detected"],
            localize=True,
            labels=True,
            style=(
                "background-color: #1a1a2e; color: #eee; font-family: monospace; "
                "font-size: 12px; padding: 10px; border-radius: 4px;"
            )
        )

        tooltip = GeoJsonTooltip(
            fields=["flood_type", "area_km2", "confidence"],
            aliases=["Type:", "Area (km²):", "Confidence:"],
            localize=True,
            sticky=False
        )

        GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip,
            popup=popup
        ).add_to(fg)

        fg.add_to(m)

    def _add_risk_zones_layer(self, m: folium.Map, geojson: Dict):
        """Add risk zone polygons (HIGH/MEDIUM/LOW) with styling."""
        fg = FeatureGroup(name="Flood Risk Zones", show=True)

        risk_colors = {
            "HIGH": {"fill": "#DC143C", "border": "#8B0000"},
            "MEDIUM": {"fill": "#FFA500", "border": "#FF6600"},
            "LOW": {"fill": "#FFD700", "border": "#DAA520"}
        }

        def style_function(feature):
            level = feature["properties"].get("risk_level", "LOW")
            colors = risk_colors.get(level, risk_colors["LOW"])
            return {
                "fillColor": colors["fill"],
                "color": colors["border"],
                "weight": 1.5,
                "fillOpacity": 0.35,
                "opacity": 0.7,
                "dashArray": "4, 4" if level == "LOW" else None
            }

        def highlight_function(feature):
            return {"weight": 2.5, "fillOpacity": 0.55}

        popup = GeoJsonPopup(
            fields=["risk_level", "risk_score", "area_km2", "population_exposed",
                    "risk_factors", "recommended_action", "confidence_method"],
            aliases=["Risk Level", "Risk Score (0-1)", "Area (km²)", "Est. Population Exposed",
                     "Risk Factors", "Recommended Action", "Analysis Method"],
            localize=True,
            labels=True,
            style=(
                "background-color: #2d1b00; color: #fff; font-family: monospace; "
                "font-size: 12px; padding: 10px; border-radius: 4px; max-width: 300px;"
            )
        )

        tooltip = GeoJsonTooltip(
            fields=["risk_level", "risk_score", "area_km2"],
            aliases=["Risk:", "Score:", "Area (km²):"],
            localize=True
        )

        GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip,
            popup=popup
        ).add_to(fg)

        fg.add_to(m)

    def _add_impact_zones_layer(self, m: folium.Map, geojson: Dict):
        """Add impact buffer zones with graduated styling."""
        fg = FeatureGroup(name="Impact Zones (Buffer Analysis)", show=False)

        impact_colors = {
            "direct_inundation": {"fill": "#8B0000", "opacity": 0.45},
            "waterlogged": {"fill": "#FF4500", "opacity": 0.30},
            "services_disrupted": {"fill": "#FF8C00", "opacity": 0.22},
            "traffic_disruption": {"fill": "#FFA500", "opacity": 0.15}
        }

        def style_function(feature):
            impact_type = feature["properties"].get("impact_type", "traffic_disruption")
            style = impact_colors.get(impact_type, impact_colors["traffic_disruption"])
            return {
                "fillColor": style["fill"],
                "color": style["fill"],
                "weight": 1,
                "fillOpacity": style["opacity"],
                "opacity": style["opacity"] + 0.2,
                "dashArray": "6, 3"
            }

        def highlight_function(feature):
            return {"weight": 2, "fillOpacity": 0.50}

        popup = GeoJsonPopup(
            fields=["impact_level", "impact_type", "buffer_distance_m", "area_km2",
                    "affected_population", "affected_hospitals", "affected_schools",
                    "affected_roads_km", "description"],
            aliases=["Impact Level", "Type", "Buffer Distance (m)", "Zone Area (km²)",
                     "Est. Population", "Hospitals", "Schools", "Roads (km)", "Description"],
            localize=True,
            labels=True
        )

        tooltip = GeoJsonTooltip(
            fields=["impact_level", "buffer_distance_m", "affected_population"],
            aliases=["Level:", "Buffer (m):", "Population:"],
            localize=True
        )

        GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip,
            popup=popup
        ).add_to(fg)

        fg.add_to(m)

    def _add_district_layer(self, m: folium.Map, geojson: Dict):
        """Add district boundaries with flood statistics."""
        fg = FeatureGroup(name="District Statistics", show=True)

        risk_border_colors = {
            "HIGH": "#DC143C",
            "MEDIUM": "#FFA500",
            "LOW": "#228B22"
        }

        def style_function(feature):
            risk = feature["properties"].get("risk_level", "LOW")
            return {
                "fillColor": "transparent",
                "color": risk_border_colors.get(risk, "#333333"),
                "weight": 2,
                "fillOpacity": 0,
                "opacity": 0.8,
                "dashArray": None
            }

        def highlight_function(feature):
            return {"weight": 3, "fillColor": "#ffffff", "fillOpacity": 0.1}

        popup = GeoJsonPopup(
            fields=["district_name", "kecamatan", "population", "district_area_km2",
                    "flood_area_km2", "flood_pct", "population_exposed", "risk_level"],
            aliases=["District", "Kecamatan", "Total Population", "District Area (km²)",
                     "Flooded Area (km²)", "Flood Coverage (%)", "Est. Population Exposed", "Risk Level"],
            localize=True,
            labels=True,
            style=(
                "background-color: #f8f9fa; color: #333; font-family: Arial; "
                "font-size: 12px; padding: 10px; border-radius: 4px;"
            )
        )

        tooltip = GeoJsonTooltip(
            fields=["district_name", "flood_area_km2", "risk_level"],
            aliases=["District:", "Flood Area (km²):", "Risk:"],
            localize=True
        )

        GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip,
            popup=popup
        ).add_to(fg)

        fg.add_to(m)

    def _add_analysis_boundary(self, m: folium.Map, bbox: tuple):
        """Add analysis area boundary."""
        min_lon, min_lat, max_lon, max_lat = bbox
        folium.Polygon(
            locations=[
                [min_lat, min_lon], [min_lat, max_lon],
                [max_lat, max_lon], [max_lat, min_lon], [min_lat, min_lon]
            ],
            color="#666666",
            fill=False,
            weight=1.5,
            dash_array="8, 4",
            tooltip="Analysis Boundary",
            name="Analysis Area"
        ).add_to(m)

    def _add_legend(self, m: folium.Map, title: str, analysis_period: str):
        """Add comprehensive map legend."""
        legend_html = """
        <div id="map-legend" style="
            position: fixed; bottom: 30px; right: 10px;
            width: 220px; background-color: rgba(255,255,255,0.95);
            border: 1px solid #ccc; border-radius: 8px;
            z-index: 9999; font-family: Arial, sans-serif;
            font-size: 12px; padding: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        ">
            <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 13px; color: #1a1a2e;">
                MAP LEGEND
            </p>

            <p style="margin: 6px 0 3px 0; font-weight: bold; color: #555; font-size: 11px;">
                FLOOD EXTENT
            </p>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#001f7a;opacity:0.75;
                    border:1px solid #00008B; margin-right:6px;"></div>
                <span>High Confidence</span>
            </div>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#0044aa;opacity:0.65;
                    border:1px solid #0000CD; margin-right:6px;"></div>
                <span>Medium Confidence</span>
            </div>

            <p style="margin: 8px 0 3px 0; font-weight: bold; color: #555; font-size: 11px;">
                FLOOD RISK ZONES
            </p>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#DC143C;opacity:0.5;
                    border:1px solid #8B0000; margin-right:6px;"></div>
                <span>HIGH Risk</span>
            </div>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#FFA500;opacity:0.4;
                    border:1px solid #FF6600; margin-right:6px;"></div>
                <span>MEDIUM Risk</span>
            </div>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#FFD700;opacity:0.35;
                    border:1px solid #DAA520; margin-right:6px;"></div>
                <span>LOW Risk</span>
            </div>

            <p style="margin: 8px 0 3px 0; font-weight: bold; color: #555; font-size: 11px;">
                IMPACT ZONES
            </p>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#8B0000;opacity:0.5;
                    border:1px dashed #8B0000; margin-right:6px;"></div>
                <span>Direct (0m)</span>
            </div>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#FF4500;opacity:0.35;
                    border:1px dashed #FF4500; margin-right:6px;"></div>
                <span>Waterlogged (500m)</span>
            </div>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#FF8C00;opacity:0.25;
                    border:1px dashed #FF8C00; margin-right:6px;"></div>
                <span>Services (1000m)</span>
            </div>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:14px;background:#FFA500;opacity:0.18;
                    border:1px dashed #FFA500; margin-right:6px;"></div>
                <span>Traffic (2000m)</span>
            </div>

            <p style="margin: 8px 0 3px 0; font-weight: bold; color: #555; font-size: 11px;">
                DISTRICTS
            </p>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:3px;background:#DC143C; margin-right:6px;"></div>
                <span>High Risk Border</span>
            </div>
            <div style="display:flex; align-items:center; margin:3px 0;">
                <div style="width:20px;height:3px;background:#FFA500; margin-right:6px;"></div>
                <span>Medium Risk Border</span>
            </div>

            <hr style="margin: 8px 0; border-color: #ddd;">
            <p style="margin: 0; color: #888; font-size: 10px;">
                Data: ESA Sentinel-1/2<br>
                Analysis: FloodLLM v1.0<br>
                Map: OpenStreetMap
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    def _add_stats_panel(self, m: folium.Map, stats: Dict[str, Any], analysis_period: str):
        """Add statistics summary panel at bottom of map."""
        flood_area = stats.get("flood_area_km2", 0)
        high_risk = stats.get("high_risk_km2", 0)
        medium_risk = stats.get("medium_risk_km2", 0)
        low_risk = stats.get("low_risk_km2", 0)
        pop_exposed = stats.get("total_population_exposed", 0)
        districts_affected = stats.get("districts_affected_count", 0)
        confidence = stats.get("confidence", "HIGH")
        updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        stats_html = f"""
        <div id="stats-panel" style="
            position: fixed; bottom: 30px; left: 10px;
            width: 260px; background-color: rgba(26,26,46,0.92);
            border: 1px solid #444; border-radius: 8px;
            z-index: 9999; font-family: Arial, sans-serif;
            color: #eee; font-size: 12px; padding: 12px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.5);
        ">
            <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 13px; color: #4fc3f7;">
                FLOOD ANALYSIS SUMMARY
            </p>
            <p style="margin: 0 0 4px 0; color: #aaa; font-size: 11px;">
                Period: {analysis_period}
            </p>
            <hr style="border-color: #444; margin: 6px 0;">
            <table style="width:100%; border-collapse:collapse;">
                <tr>
                    <td style="padding:3px 0; color:#aaa;">Total Flood Area:</td>
                    <td style="padding:3px 0; text-align:right; font-weight:bold; color:#4fc3f7;">
                        {flood_area:.2f} km²
                    </td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#aaa;">Est. Pop. Exposed:</td>
                    <td style="padding:3px 0; text-align:right; font-weight:bold; color:#ff9800;">
                        {pop_exposed:,}
                    </td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#aaa;">High Risk Area:</td>
                    <td style="padding:3px 0; text-align:right; color:#ef5350;">
                        {high_risk:.2f} km²
                    </td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#aaa;">Medium Risk Area:</td>
                    <td style="padding:3px 0; text-align:right; color:#ffa726;">
                        {medium_risk:.2f} km²
                    </td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#aaa;">Low Risk Area:</td>
                    <td style="padding:3px 0; text-align:right; color:#ffee58;">
                        {low_risk:.2f} km²
                    </td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#aaa;">Districts Affected:</td>
                    <td style="padding:3px 0; text-align:right; color:#aaa;">
                        {districts_affected}
                    </td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#aaa;">Confidence:</td>
                    <td style="padding:3px 0; text-align:right; color:#66bb6a;">
                        {confidence}
                    </td>
                </tr>
            </table>
            <hr style="border-color: #444; margin: 6px 0;">
            <p style="margin: 0; color: #666; font-size: 10px;">
                Updated: {updated}
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(stats_html))

    def _add_title_panel(self, m: folium.Map, title: str, analysis_period: str):
        """Add title panel at top of map."""
        title_html = f"""
        <div style="
            position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
            background-color: rgba(26,26,46,0.90); color: white;
            padding: 8px 18px; border-radius: 6px; z-index: 9999;
            font-family: Arial, sans-serif; font-size: 14px; font-weight: bold;
            border: 1px solid #4fc3f7; white-space: nowrap;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        ">
            {title} — {analysis_period}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
