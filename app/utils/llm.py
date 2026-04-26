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
- "location_name": (string) City/region name, append ', Indonesia' if clearly in Indonesia. E.g. 'Semarang, Indonesia'
- "start_date": (string) Start date in YYYY-MM-DD format (absolute, resolved from relative expression)
- "end_date": (string) End date in YYYY-MM-DD format

Date resolution rules (in priority order):
1. If a 4-digit year is mentioned (e.g. "2020", "tahun 2020", "pada 2020"):
   start_date = YYYY-01-01, end_date = YYYY-12-31
2. If a month + year is mentioned (e.g. "Januari 2020", "January 2020", "Februari 2024"):
   start_date = YYYY-MM-01, end_date = last day of that month
3. If relative Indonesian expressions are used:
   - 'N hari kebelakang/yang lalu' = N days ago from today
   - 'N minggu kebelakang/yang lalu' = N weeks ago
   - 'N bulan kebelakang/yang lalu' = N months ago (approx 30 days)
   - 'N tahun kebelakang/yang lalu' = N years ago (approx 365 days)
   - 'kemarin' = yesterday, 'minggu lalu' = 7 days ago, 'bulan lalu' = 30 days ago
4. If relative English expressions: 'last N days/weeks/months', 'N days ago', etc.
5. Default: last 7 days

Location extraction rules:
- Look for the main geographic entity in the query (city, region, country)
- Common Indonesian prefixes to ignore: 'di', 'untuk', 'daerah', 'wilayah', 'kota', 'kawasan'
- The location may appear directly after flood-related keywords: 'banjir [city]', 'flood [city]', 'analisis [city]'
- Examples: 'banjir Jakarta', 'analisis risiko Jakarta', 'flood Semarang', 'banjir di Demak'

Examples:
Input: 'Today is 2026-04-26.\n\nanalisis banjir Jakarta 2020'
Output: {"location_name": "Jakarta, Indonesia", "start_date": "2020-01-01", "end_date": "2020-12-31"}

Input: 'Today is 2026-04-26.\n\nanalisis risiko banjir di Jakarta pada tahun 2020'
Output: {"location_name": "Jakarta, Indonesia", "start_date": "2020-01-01", "end_date": "2020-12-31"}

Input: 'Today is 2026-04-26.\n\nbanjir Semarang Februari 2020'
Output: {"location_name": "Semarang, Indonesia", "start_date": "2020-02-01", "end_date": "2020-02-29"}

Input: 'Today is 2026-04-26.\n\nflood analysis in Bandung January 2021'
Output: {"location_name": "Bandung, Indonesia", "start_date": "2021-01-01", "end_date": "2021-01-31"}

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


# Indonesian month name → (month_number, days_in_month_non_leap)
_ID_MONTHS = {
    'januari': (1, 31), 'februari': (2, 28), 'maret': (3, 31),
    'april': (4, 30), 'mei': (5, 31), 'juni': (6, 30),
    'juli': (7, 31), 'agustus': (8, 31), 'september': (9, 30),
    'oktober': (10, 31), 'november': (11, 30), 'desember': (12, 31),
    'january': (1, 31), 'february': (2, 28), 'march': (3, 31),
    'april': (4, 30), 'may': (5, 31), 'june': (6, 30),
    'july': (7, 31), 'august': (8, 31), 'september': (9, 30),
    'october': (10, 31), 'november': (11, 30), 'december': (12, 31),
}

# Words that terminate a city name scan
_STOP_WORDS = {
    'pada', 'selama', 'dalam', 'untuk', 'for', 'during', 'since',
    'minggu', 'bulan', 'hari', 'tahun', 'kemarin', 'lalu', 'kebelakang',
    'yang', 'last', 'ago', 'past', 'the', 'and', 'or', 'a', 'an',
    'analisis', 'analisa', 'analysis', 'risiko', 'risk', 'deteksi',
    'detection', 'banjir', 'flood', 'genangan', 'monitoring', 'assessment',
    'show', 'tampilkan', 'lihat', 'cek', 'check', 'laporan', 'report',
}


def _days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


class LLMPromptHandler:
    """Handle LLM-based prompt parsing and report generation."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize LLM handler."""
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

    def parse_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Parse natural language prompt into structured query via Gemini."""
        if not self.model:
            return self._simple_parse(user_prompt)

        messages = get_parsing_messages(user_prompt)
        try:
            parse_model = genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=messages[0]["content"]
            )
            response = parse_model.generate_content(messages[1]["content"])
            raw = response.text.strip()
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if fenced:
                raw = fenced.group(1)
            parsed = json.loads(raw)
            parsed["original_prompt"] = user_prompt
            return parsed
        except Exception as e:
            print(f"LLM parsing error: {e}")

        return self._simple_parse(user_prompt)

    def _simple_parse(self, user_prompt: str) -> Dict[str, Any]:
        """Deterministic fallback parser — handles common Indonesian + English patterns."""
        import re as _re
        from datetime import datetime as _dt, timedelta as _td
        import calendar as _cal

        now = _dt.now()
        today_str = now.strftime('%Y-%m-%d')
        text = user_prompt

        # ── Location extraction ───────────────────────────────────────────
        # Month names should not be included in city names
        _MONTH_NAMES = set(_ID_MONTHS.keys())

        def _collect_city(rest: str) -> Optional[str]:
            """Collect city words from the start of `rest`, stopping at stop words/digits."""
            city_words = []
            for word in rest.split():
                clean = word.strip('.,!?;:')
                if not clean:
                    continue
                if clean.lower() in _STOP_WORDS or clean.lower() in _MONTH_NAMES:
                    break
                if clean[:1].isdigit():
                    break
                city_words.append(clean)
                if len(city_words) >= 4:
                    break
            return ' '.join(city_words) if city_words else None

        def _extract_city(t: str) -> Optional[str]:
            # 1. Try explicit location prefix (highest confidence)
            m = _re.search(
                r'\b(?:in|at|di|untuk|daerah|wilayah|kota|kawasan)\s+', t, _re.IGNORECASE
            )
            if m:
                result = _collect_city(t[m.end():])
                if result:
                    return result

            # 2. Try all flood-related keywords and take the first valid city found
            #    Use finditer to get all matches; try from the last match backwards
            #    (e.g. "analisis banjir Jakarta" → try after "banjir" first)
            flood_kw_pat = _re.compile(
                r'\b(?:banjir|flood|genangan|deteksi|analisis|analysis|risiko|risk|laporan|report)\s+',
                _re.IGNORECASE
            )
            matches = list(flood_kw_pat.finditer(t))
            # Try from the last match: more likely to be closest to the city name
            for m2 in reversed(matches):
                result = _collect_city(t[m2.end():])
                if result:
                    return result

            return None

        location_name = _extract_city(text) or 'unknown'

        # ── Date extraction ───────────────────────────────────────────────
        start_dt: Optional[_dt] = None
        end_dt: Optional[_dt] = None

        # 1. Month + year (e.g. "Januari 2020", "February 2024")
        for month_name, (month_num, _) in _ID_MONTHS.items():
            pat = _re.compile(
                r'\b' + month_name + r'\s+(20\d{2})\b', _re.IGNORECASE
            )
            m = pat.search(text)
            if m:
                yr = int(m.group(1))
                days = _days_in_month(yr, month_num)
                start_dt = _dt(yr, month_num, 1)
                end_dt = _dt(yr, month_num, days)
                break

        # 2. Plain 4-digit year (e.g. "tahun 2020", "pada 2020", or just "2020")
        if start_dt is None:
            m = _re.search(r'\b(20\d{2})\b', text)
            if m:
                yr = int(m.group(1))
                start_dt = _dt(yr, 1, 1)
                end_dt = _dt(yr, 12, 31)

        # 3. Indonesian numeric relative patterns
        if start_dt is None:
            for pat, days_mult in [
                (r'(\d+)\s*tahun', 365),
                (r'(\d+)\s*bulan', 30),
                (r'(\d+)\s*minggu', 7),
                (r'(\d+)\s*hari', 1),
            ]:
                m = _re.search(pat, text, _re.IGNORECASE)
                if m:
                    start_dt = now - _td(days=int(m.group(1)) * days_mult)
                    break

        # 4. Indonesian keyword relative patterns
        if start_dt is None:
            if _re.search(r'tahun\s*lalu', text, _re.IGNORECASE):
                start_dt = now - _td(days=365)
            elif _re.search(r'bulan\s*lalu', text, _re.IGNORECASE):
                start_dt = now - _td(days=30)
            elif _re.search(r'minggu\s*lalu', text, _re.IGNORECASE):
                start_dt = now - _td(days=7)
            elif _re.search(r'kemarin', text, _re.IGNORECASE):
                start_dt = now - _td(days=1)

        # 5. English relative patterns
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
                m = _re.search(pat, text, _re.IGNORECASE)
                if m:
                    start_dt = now - _td(days=int(m.group(1)) * days_mult)
                    break

        # 6. Explicit YYYY-MM-DD
        if start_dt is None:
            m = _re.search(r'(\d{4}-\d{2}-\d{2})', text)
            if m:
                try:
                    start_dt = _dt.strptime(m.group(1), '%Y-%m-%d')
                except ValueError:
                    pass

        # 7. Default: last 7 days
        if start_dt is None:
            start_dt = now - _td(days=7)

        if end_dt is None:
            end_dt = now

        return {
            'location_name': location_name,
            'start_date': start_dt.strftime('%Y-%m-%d'),
            'end_date': end_dt.strftime('%Y-%m-%d'),
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

{f"Rainfall (analysis period): {rainfall_data.get('total_mm', 0):.1f} mm" if rainfall_data else ""}

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
