"""Geocoding and location utilities."""
import httpx
from typing import Optional, Tuple
import math

# Maximum buffer around city center (degrees). ~0.4° ≈ 44 km radius.
# Prevents Nominatim's bounding boxes from including remote outlying territories
# (e.g. Jakarta's Kepulauan Seribu islands 150 km north of the city core).
_MAX_BUFFER_DEG = 0.40


async def geocode_location(location: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Geocode a location name to bounding box.

    Returns: (min_lon, min_lat, max_lon, max_lat) or None
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location,
        "format": "json",
        "limit": 1,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": "flood-llm/1.0"},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            if data:
                result = data[0]
                center_lat = float(result["lat"])
                center_lon = float(result["lon"])

                # Nominatim boundingbox: [south, north, west, east]
                raw_bb = result.get("boundingbox")
                if raw_bb and len(raw_bb) == 4:
                    min_lat = float(raw_bb[0])
                    max_lat = float(raw_bb[1])
                    min_lon = float(raw_bb[2])
                    max_lon = float(raw_bb[3])
                else:
                    # Fallback: use center + buffer
                    min_lon = center_lon - _MAX_BUFFER_DEG
                    max_lon = center_lon + _MAX_BUFFER_DEG
                    min_lat = center_lat - _MAX_BUFFER_DEG
                    max_lat = center_lat + _MAX_BUFFER_DEG

                # Clamp bbox to _MAX_BUFFER_DEG around the geocoded center so
                # outlying territories don't inflate the analysis area.
                min_lon = max(min_lon, center_lon - _MAX_BUFFER_DEG)
                max_lon = min(max_lon, center_lon + _MAX_BUFFER_DEG)
                min_lat = max(min_lat, center_lat - _MAX_BUFFER_DEG)
                max_lat = min(max_lat, center_lat + _MAX_BUFFER_DEG)

                return (
                    round(min_lon, 6),
                    round(min_lat, 6),
                    round(max_lon, 6),
                    round(max_lat, 6),
                )

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
                headers={"User-Agent": "flood-llm/1.0"},
                timeout=10.0,
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

    km_per_deg_lat = 111.32
    center_lat_rad = ((min_lat + max_lat) / 2) * math.pi / 180
    km_per_deg_lon = 111.32 * abs(math.cos(center_lat_rad))

    width_km = (max_lon - min_lon) * km_per_deg_lon
    height_km = (max_lat - min_lat) * km_per_deg_lat

    return width_km * height_km
