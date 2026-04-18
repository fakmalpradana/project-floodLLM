"""LLM integration for prompt parsing and report generation."""
import google.generativeai as genai
from typing import Dict, Any, Optional
from ..utils.config import settings
import json
import re
from datetime import datetime, timedelta

SYSTEM_PROMPT = """You are an expert Geospatial AI Assistant for the FloodLLM system that handles queries in BOTH English AND Indonesian (Bahasa Indonesia).

Your task is to parse user queries about flood events and extract spatio-temporal parameters.
You MUST output ONLY a strict, valid JSON object with NO markdown fences, no explanation, no extra text.

The user message will start with 'Today is YYYY-MM-DD.' — use this date to resolve ALL relative time expressions.

Required output fields:
- "location_name": (string) City/region name, append ', Indonesia' if in Indonesia. E.g. 'Semarang, Indonesia'
- "start_date": (string) Start date in YYYY-MM-DD format (absolute, resolved from relative expression using today's date)
- "end_date": (string) End date in YYYY-MM-DD format

Indonesian time keywords to handle:
- 'N hari kebelakang/yang lalu' = N days ago
- 'N minggu kebelakang/yang lalu' = N weeks ago
- 'N bulan kebelakang/yang lalu' = N months ago (approx 30 days)
- 'N tahun kebelakang/yang lalu' = N years ago (approx 365 days)
- 'kemarin' = yesterday, 'minggu lalu' = 7 days ago, 'bulan lalu' = 30 days ago, 'tahun lalu' = 365 days ago
- Indonesian location prefix: 'di [city]', 'untuk [city]', 'daerah [city]'

Example:
Input: 'Today is 2025-04-18.\n\nanalisis banjir di surabaya pada 1 tahun kebelakang'
Output: {"location_name": "Surabaya, Indonesia", "start_date": "2024-04-18", "end_date": "2025-04-18"}
"""


def get_parsing_messages(user_query: str) -> list[dict[str, str]]:
    from datetime import datetime as _dt
    today = _dt.now().strftime('%Y-%m-%d')
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Today is {today}.\n\n{user_query}"}
    ]


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
        import re as _re
        from datetime import datetime as _dt, timedelta as _td
        now = _dt.now()
        today_str = now.strftime('%Y-%m-%d')

        # Closed set of stop words that terminate a city name
        _STOP = {
            'pada', 'selama', 'dalam', 'untuk', 'for', 'during', 'since',
            'minggu', 'bulan', 'hari', 'tahun', 'kemarin', 'lalu', 'kebelakang',
            'yang', 'last', 'ago', 'past', 'the', 'and', 'or', 'a', 'an',
        }

        def _extract_city(text: str) -> str | None:
            """Find the location prefix then collect words until a stop word or digit."""
            # Find the location-introducing keyword
            m = _re.search(
                r'\b(?:in|at|di|untuk|daerah|wilayah)\s+', text, _re.IGNORECASE
            )
            if not m:
                return None
            rest = text[m.end():]
            city_words = []
            for word in rest.split():
                clean = word.strip('.,!?')
                if clean.lower() in _STOP or clean[:1].isdigit():
                    break
                city_words.append(clean)
                if len(city_words) >= 4:   # cap at 4-word city names
                    break
            return ' '.join(city_words) if city_words else None

        location_name = _extract_city(user_prompt) or 'unknown'

        # Resolve date — try Indonesian then English patterns
        start_dt = None

        # Indonesian numeric patterns
        for pat, days_mult in [
            (r'(\d+)\s*tahun', 365),
            (r'(\d+)\s*bulan', 30),
            (r'(\d+)\s*minggu', 7),
            (r'(\d+)\s*hari', 1),
        ]:
            m = _re.search(pat, user_prompt, _re.IGNORECASE)
            if m:
                start_dt = now - _td(days=int(m.group(1)) * days_mult)
                break

        # Indonesian keyword patterns
        if start_dt is None:
            if _re.search(r'tahun\s*lalu', user_prompt, _re.IGNORECASE):
                start_dt = now - _td(days=365)
            elif _re.search(r'bulan\s*lalu', user_prompt, _re.IGNORECASE):
                start_dt = now - _td(days=30)
            elif _re.search(r'minggu\s*lalu', user_prompt, _re.IGNORECASE):
                start_dt = now - _td(days=7)
            elif _re.search(r'kemarin', user_prompt, _re.IGNORECASE):
                start_dt = now - _td(days=1)

        # English numeric patterns
        if start_dt is None:
            for pat, days_mult in [
                (r'last\s+(\d+)\s*year', 365),
                (r'last\s+(\d+)\s*month', 30),
                (r'last\s+(\d+)\s*week', 7),
                (r'last\s+(\d+)\s*day', 1),
                (r'(\d+)\s*years?\s*ago', 365),
                (r'(\d+)\s*months?\s*ago', 30),
                (r'(\d+)\s*weeks?\s*ago', 7),
                (r'(\d+)\s*days?\s*ago', 1),
            ]:
                m = _re.search(pat, user_prompt, _re.IGNORECASE)
                if m:
                    start_dt = now - _td(days=int(m.group(1)) * days_mult)
                    break

        # YYYY-MM-DD literal in prompt
        if start_dt is None:
            m = _re.search(r'(\d{4}-\d{2}-\d{2})', user_prompt)
            if m:
                try:
                    start_dt = _dt.strptime(m.group(1), '%Y-%m-%d')
                except ValueError:
                    pass

        if start_dt is None:
            start_dt = now - _td(days=7)  # default: last 7 days

        return {
            'location_name': location_name,
            'start_date': start_dt.strftime('%Y-%m-%d'),
            'end_date': today_str,
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
