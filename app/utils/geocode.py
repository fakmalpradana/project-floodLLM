"""Geocoding and location utilities."""
import httpx
from typing import Optional, Tuple


async def geocode_location(location: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Geocode a location name to bounding box.

    Returns: (min_lon, min_lat, max_lon, max_lat) or None
    """
    # Using Nominatim (OpenStreetMap) - free, no API key needed
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location,
        "format": "json",
        "limit": 1,
        "polygon_geojson": 1
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": "flood-llm/1.0"}
            )
            response.raise_for_status()
            data = response.json()

            if data:
                result = data[0]
                if "geojson" in result:
                    bbox = result["geojson"].get("bbox")
                    if bbox:
                        return tuple(bbox)

                # Fallback to bounding box from lat/lon + buffer
                lat = float(result["lat"])
                lon = float(result["lon"])
                buffer = 0.5  # ~50km
                return (lon - buffer, lat - buffer, lon + buffer, lat + buffer)

        except Exception as e:
            print(f"Geocoding error: {e}")

    return None


async def reverse_geocode(lat: float, lon: float) -> str:
    """Reverse geocode coordinates to location name."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": "flood-llm/1.0"}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("display_name", f"{lat:.4f}, {lon:.4f}")
        except Exception as e:
            print(f"Reverse geocoding error: {e}")
            return f"{lat:.4f}, {lon:.4f}"


def calculate_bbox_area_km2(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate approximate area of bounding box in km²."""
    min_lon, min_lat, max_lon, max_lat = bbox

    # Approximate km per degree
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * abs(((min_lat + max_lat) / 2) * 3.14159 / 180)

    width_km = (max_lon - min_lon) * km_per_deg_lon
    height_km = (max_lat - min_lat) * km_per_deg_lat

    return width_km * height_km
