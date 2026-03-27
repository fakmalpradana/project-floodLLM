"""LLM integration for prompt parsing and report generation."""
import google.generativeai as genai
from typing import Dict, Any, Optional
from ..utils.config import settings
import json
import re


class LLMPromptHandler:
    """Handle LLM-based prompt parsing and report generation."""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize LLM handler."""
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

    def parse_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """
        Parse natural language prompt into structured query.

        Example input: "Show flood extent in Jakarta for the last 7 days"
        Returns: {
            "location": "Jakarta, Indonesia",
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "date_start": "2024-01-01",
            "date_end": "2024-01-08",
            "task_type": "flood_detection",
            "original_prompt": "..."
        }
        """
        if not self.model:
            # Fallback: simple parsing
            return self._simple_parse(user_prompt)

        prompt = f"""
You are a flood monitoring assistant. Parse this user request about flood detection:

"{user_prompt}"

Extract the following information as JSON:
- location: The place name (city, region, country)
- date_range: Start and end dates (use "last 7 days" if not specified)
- task_type: One of "flood_detection", "risk_prediction", "damage_assessment", "all"
- additional_context: Any other relevant details

Respond ONLY with valid JSON in this format:
{{
    "location": "place name",
    "date_start": "YYYY-MM-DD or relative like 'last 7 days'",
    "date_end": "YYYY-MM-DD or 'today'",
    "task_type": "flood_detection|risk_prediction|damage_assessment|all",
    "additional_context": "any extra details"
}}
"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                parsed["original_prompt"] = user_prompt
                return parsed
        except Exception as e:
            print(f"LLM parsing error: {e}")

        return self._simple_parse(user_prompt)

    def _simple_parse(self, user_prompt: str) -> Dict[str, Any]:
        """Simple regex-based fallback parsing."""
        # Extract location ( crude but works)
        location_match = re.search(r'in ([A-Za-z\s,]+?)(?:\s*(?:for|during|since|$))', user_prompt)
        location = location_match.group(1).strip() if location_match else "unknown"

        # Extract date range
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|until|-)?\s*(\d{4}-\d{2}-\d{2})?', user_prompt)
        if date_match:
            date_start = date_match.group(1)
            date_end = date_match.group(2) or "today"
        else:
            date_start = "last 7 days"
            date_end = "today"

        # Determine task type
        if "risk" in user_prompt.lower():
            task_type = "risk_prediction"
        elif "damage" in user_prompt.lower() or "assessment" in user_prompt.lower():
            task_type = "damage_assessment"
        elif "map" in user_prompt.lower() or "extent" in user_prompt.lower() or "flood" in user_prompt.lower():
            task_type = "flood_detection"
        else:
            task_type = "all"

        return {
            "location": location,
            "date_start": date_start,
            "date_end": date_end,
            "task_type": task_type,
            "original_prompt": user_prompt
        }

    def generate_report(
        self,
        location: str,
        date_range: str,
        flood_area_km2: float,
        affected_infrastructure: Dict[str, int],
        rainfall_data: Optional[Dict] = None
    ) -> str:
        """Generate a natural language flood report."""

        if not self.model:
            return self._simple_report(location, date_range, flood_area_km2, affected_infrastructure)

        prompt = f"""
Generate a concise flood assessment report based on the following data:

Location: {location}
Date Range: {date_range}
Estimated Flood Area: {flood_area_km2:.2f} km²
Affected Infrastructure:
- Buildings: {affected_infrastructure.get('buildings', 0)}
- Roads: {affected_infrastructure.get('roads_km', 0)} km
- Agricultural Land: {affected_infrastructure.get('agricultural_km2', 0):.2f} km²

{f"Rainfall (last 7 days): {rainfall_data.get('total_mm', 0):.1f} mm" if rainfall_data else ""}

Write a professional 2-3 paragraph assessment suitable for emergency response coordination.
Include severity assessment and recommended actions.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Report generation error: {e}")
            return self._simple_report(location, date_range, flood_area_km2, affected_infrastructure)

    def _simple_report(
        self,
        location: str,
        date_range: str,
        flood_area_km2: float,
        affected_infrastructure: Dict[str, int]
    ) -> str:
        """Simple template-based report."""
        severity = "severe" if flood_area_km2 > 100 else "moderate" if flood_area_km2 > 10 else "minor"

        return f"""
FLOOD ASSESSMENT REPORT
=======================

Location: {location}
Period: {date_range}
Severity: {severity.upper()}

SUMMARY
-------
Satellite analysis has detected approximately {flood_area_km2:.2f} km² of flooded area.

AFFECTED INFRASTRUCTURE
-----------------------
- Buildings in flood zone: ~{affected_infrastructure.get('buildings', 0)}
- Roads potentially affected: ~{affected_infrastructure.get('roads_km', 0)} km
- Agricultural land: ~{affected_infrastructure.get('agricultural_km2', 0):.2f} km²

RECOMMENDATIONS
---------------
1. Prioritize evacuation of low-lying areas
2. Deploy emergency supplies to affected zones
3. Monitor water levels and rainfall forecasts
4. Coordinate with local emergency services

Report generated by FloodLLM automated system.
"""
