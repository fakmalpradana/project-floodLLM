#!/usr/bin/env python3
"""
Phase 3 Test Runner: Jakarta January 2025 Flood Analysis
=========================================================
Executes the full satellite-based vector layer analysis pipeline:
  1. Change detection (S1+S2 fusion simulation)
  2. Flood extent vector generation (GeoJSON polygons)
  3. Flood risk zone classification (HIGH/MEDIUM/LOW)
  4. Impact buffer zones (0/500/1000/2000m)
  5. District-level statistics
  6. Interactive vector map (Folium)
  7. Comprehensive satellite analysis report (HTML)

Usage:
    python run_phase3.py
"""

import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))


def print_section(title: str):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_check(label: str, passed: bool, detail: str = ""):
    status = "✓" if passed else "✗"
    color_start = "" if passed else ""
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {status} {label}{detail_str}")


def run_analysis():
    print("=" * 60)
    print("  FLOODLLM PHASE 3 — VECTOR LAYER SATELLITE ANALYSIS")
    print("  Jakarta, Indonesia — January 2025")
    print("=" * 60)

    start_time = time.time()

    # Configuration
    LOCATION = "Jakarta, Indonesia"
    ANALYSIS_PERIOD = "January 2025"
    BBOX = (106.65, -6.37, 107.00, -6.05)  # Jakarta bounding box
    JOB_ID = f"jakarta_2025_{str(uuid.uuid4())[:6]}"
    RAINFALL_MM = 185.3  # January 2025 rainfall estimate (mm)

    print(f"\n  Location : {LOCATION}")
    print(f"  Period   : {ANALYSIS_PERIOD}")
    print(f"  BBox     : {BBOX}")
    print(f"  Job ID   : {JOB_ID}")
    print(f"  Rainfall : {RAINFALL_MM} mm")

    results = {}

    # ── Step 1: Change Detection ───────────────────────────────────────────
    print_section("STEP 1: Change Detection (S1+S2 Fusion)")

    try:
        from app.processing.change_detection import ChangeDetector
        detector = ChangeDetector()

        # Simulate change detection (no real satellite data available in test)
        change_result = detector.compute_flood_change(
            baseline_ndwi=None,
            flood_ndwi=None,
            bbox=BBOX
        )

        severity = detector.compute_flood_severity(
            flood_area_km2=change_result.get("new_flood_area_km2", 10.2)
        )

        results["change_detection"] = change_result
        results["severity"] = severity

        fusion = change_result.get("fusion", {})
        flood_km2 = change_result.get("new_flood_area_km2", 10.2)

        print_check("Change detection completed", True)
        print_check(f"Method: {fusion.get('method', 'N/A')}", True)
        print_check(
            f"New flood area: {flood_km2:.3f} km²",
            flood_km2 > 0.01,
            f"{flood_km2 * 100:.0f} ha"
        )
        print_check(
            f"Sensor agreement: {fusion.get('agreement_rate_pct', 0):.1f}%",
            fusion.get("agreement_rate_pct", 0) > 70
        )
        print_check(
            f"Confidence: {fusion.get('confidence', 'N/A')}",
            fusion.get("confidence") in ("HIGH", "MEDIUM")
        )
        print_check(
            f"Severity: {severity.get('severity', 'N/A')} ({severity.get('flood_pct', 0):.2f}% of Jakarta)",
            True
        )

    except Exception as e:
        print(f"  ✗ Change detection failed: {e}")
        import traceback
        traceback.print_exc()
        results["change_detection"] = {}
        return

    # ── Step 2: Flood Extent Vector Generation ────────────────────────────
    print_section("STEP 2: Flood Extent Vector Generation")

    try:
        from app.processing.vector_generator import VectorGenerator
        vector_gen = VectorGenerator()

        flood_extent_result = vector_gen.generate_flood_extent_vector(
            flood_mask=None,  # Triggers Jakarta simulation
            bbox=BBOX,
            job_id=JOB_ID,
            source="Sentinel-1 SAR + Sentinel-2 Optical",
            confidence="HIGH",
            date_detected="2025-01-15"
        )

        results["flood_extent"] = flood_extent_result

        n_features = flood_extent_result.get("feature_count", 0)
        total_area = flood_extent_result.get("total_area_km2", 0)
        path = flood_extent_result.get("path")

        print_check(f"Flood extent polygons generated: {n_features}", n_features >= 1)
        print_check(f"Total flood area: {total_area:.3f} km²", total_area > 0.01)
        print_check(f"GeoJSON saved: {Path(path).name if path else 'N/A'}", path is not None)

        # Validate GeoJSON geometry
        geojson = flood_extent_result.get("geojson", {})
        valid_geom = all(
            f.get("geometry") is not None
            for f in geojson.get("features", [])
        )
        print_check("All geometries valid", valid_geom)

    except Exception as e:
        print(f"  ✗ Flood extent generation failed: {e}")
        import traceback
        traceback.print_exc()
        results["flood_extent"] = {"geojson": {"features": []}, "total_area_km2": 0}

    # ── Step 3: Risk Zone Generation ──────────────────────────────────────
    print_section("STEP 3: Flood Risk Zone Generation")

    try:
        risk_zones_result = vector_gen.generate_risk_zones(
            bbox=BBOX,
            risk_map=None,
            flood_extent_geojson=results["flood_extent"].get("geojson"),
            job_id=JOB_ID
        )

        results["risk_zones"] = risk_zones_result

        n_features = risk_zones_result.get("feature_count", 0)
        high_km2 = risk_zones_result.get("high_risk_km2", 0)
        medium_km2 = risk_zones_result.get("medium_risk_km2", 0)
        low_km2 = risk_zones_result.get("low_risk_km2", 0)

        # Count by level
        features = risk_zones_result.get("geojson", {}).get("features", [])
        n_high = sum(1 for f in features if f["properties"].get("risk_level") == "HIGH")
        n_medium = sum(1 for f in features if f["properties"].get("risk_level") == "MEDIUM")
        n_low = sum(1 for f in features if f["properties"].get("risk_level") == "LOW")

        print_check(f"Risk zone polygons: {n_features}", n_features >= 1)
        print_check(
            f"HIGH risk zones: {n_high} ({high_km2:.2f} km²)",
            n_high >= 1,
        )
        print_check(
            f"MEDIUM risk zones: {n_medium} ({medium_km2:.2f} km²)",
            n_medium >= 1,
        )
        print_check(
            f"LOW risk zones: {n_low} ({low_km2:.2f} km²)",
            n_low >= 1,
        )
        print_check(
            f"Risk zones saved: {Path(risk_zones_result.get('path', '')).name if risk_zones_result.get('path') else 'N/A'}",
            risk_zones_result.get("path") is not None
        )

    except Exception as e:
        print(f"  ✗ Risk zone generation failed: {e}")
        import traceback
        traceback.print_exc()
        results["risk_zones"] = {"geojson": {"features": []}}

    # ── Step 4: Impact Buffer Zones ───────────────────────────────────────
    print_section("STEP 4: Impact Buffer Zones (0/500/1000/2000m)")

    try:
        impact_zones_result = vector_gen.generate_impact_zones(
            flood_extent_geojson=results["flood_extent"].get("geojson"),
            job_id=JOB_ID,
            date_analysis="2025-01-15"
        )

        results["impact_zones"] = impact_zones_result

        n_zones = impact_zones_result.get("feature_count", 0)
        zones = impact_zones_result.get("zones", {})
        total_pop = impact_zones_result.get("total_affected_population", 0)

        print_check(f"Impact zones generated: {n_zones}", n_zones >= 1)
        print_check(
            f"Direct inundation zone: {zones.get('direct_inundation_km2', 0):.3f} km²",
            zones.get("direct_inundation_km2", 0) > 0
        )
        print_check(
            f"500m buffer zone: {zones.get('500m_buffer_km2', 0):.3f} km²",
            zones.get("500m_buffer_km2", 0) > 0
        )
        print_check(
            f"1000m buffer zone: {zones.get('1000m_buffer_km2', 0):.3f} km²",
            zones.get("1000m_buffer_km2", 0) > 0
        )
        print_check(
            f"2000m buffer zone: {zones.get('2000m_buffer_km2', 0):.3f} km²",
            zones.get("2000m_buffer_km2", 0) > 0
        )
        print_check(f"Total affected population: ~{total_pop:,}", total_pop > 0)

    except Exception as e:
        print(f"  ✗ Impact zone generation failed: {e}")
        import traceback
        traceback.print_exc()
        results["impact_zones"] = {"geojson": {"features": []}, "zones": {}}

    # ── Step 5: District Statistics ───────────────────────────────────────
    print_section("STEP 5: District-Level Statistics")

    try:
        districts_result = vector_gen.generate_district_statistics(
            flood_extent_geojson=results["flood_extent"].get("geojson"),
            risk_zones_geojson=results["risk_zones"].get("geojson"),
            bbox=BBOX,
            job_id=JOB_ID
        )

        results["districts"] = districts_result

        n_districts = districts_result.get("feature_count", 0)
        total_flood = districts_result.get("total_flood_area_km2", 0)
        total_pop_exposed = districts_result.get("total_population_exposed", 0)
        affected = districts_result.get("affected_districts", [])

        print_check(f"Districts analyzed: {n_districts}", n_districts >= 5)
        print_check(f"Total flood area: {total_flood:.3f} km²", total_flood > 0)
        print_check(f"Total population exposed: {total_pop_exposed:,}", total_pop_exposed > 0)
        print_check(f"Districts with flooding: {len(affected)}", len(affected) >= 1)
        if affected:
            print(f"      Affected: {', '.join(affected[:4])}{'...' if len(affected) > 4 else ''}")

    except Exception as e:
        print(f"  ✗ District statistics failed: {e}")
        import traceback
        traceback.print_exc()
        results["districts"] = {"geojson": {"features": []}}

    # ── Step 6: Interactive Vector Map ────────────────────────────────────
    print_section("STEP 6: Interactive Vector Map")

    try:
        from app.visualization.vector_map import VectorFloodMap
        vector_mapper = VectorFloodMap()

        analysis_stats = {
            "flood_area_km2": results["flood_extent"].get("total_area_km2", 0),
            "high_risk_km2": results["risk_zones"].get("high_risk_km2", 0),
            "medium_risk_km2": results["risk_zones"].get("medium_risk_km2", 0),
            "low_risk_km2": results["risk_zones"].get("low_risk_km2", 0),
            "total_population_exposed": results["districts"].get("total_population_exposed", 0),
            "districts_affected_count": results["districts"].get("district_count_affected", 0),
            "confidence": results["change_detection"].get("fusion", {}).get("confidence", "HIGH")
        }

        map_result = vector_mapper.create_vector_map(
            job_id=JOB_ID,
            bbox=BBOX,
            flood_extent_geojson=results["flood_extent"].get("geojson"),
            risk_zones_geojson=results["risk_zones"].get("geojson"),
            impact_zones_geojson=results["impact_zones"].get("geojson"),
            districts_geojson=results["districts"].get("geojson"),
            analysis_stats=analysis_stats,
            title=f"Flood Risk & Impact Assessment — {LOCATION}",
            analysis_period=ANALYSIS_PERIOD
        )

        results["map"] = map_result

        map_path = map_result.get("map_path")
        has_error = "error" in map_result and map_result["error"]
        layers = map_result.get("layers", {})

        print_check("Map created successfully", not has_error and map_path is not None)
        if has_error:
            print(f"      Error: {map_result['error']}")
        if map_path:
            print_check(f"Map saved: {Path(map_path).name}", Path(map_path).exists())
            print_check("Flood extent layer", layers.get("flood_extent", False))
            print_check("Risk zones layer", layers.get("risk_zones", False))
            print_check("Impact zones layer", layers.get("impact_zones", False))
            print_check("Districts layer", layers.get("districts", False))

    except Exception as e:
        print(f"  ✗ Map generation failed: {e}")
        import traceback
        traceback.print_exc()
        results["map"] = {}

    # ── Step 7: Satellite Analysis Report ────────────────────────────────
    print_section("STEP 7: Comprehensive Satellite Analysis Report")

    try:
        from app.visualization.satellite_report import SatelliteFloodReport
        reporter = SatelliteFloodReport()

        report_path = reporter.generate(
            job_id=JOB_ID,
            location=LOCATION,
            analysis_period=ANALYSIS_PERIOD,
            bbox=BBOX,
            flood_extent_result=results.get("flood_extent"),
            risk_zones_result=results.get("risk_zones"),
            impact_zones_result=results.get("impact_zones"),
            districts_result=results.get("districts"),
            change_detection_result=results.get("change_detection"),
            map_path=results.get("map", {}).get("map_path"),
            rainfall_mm=RAINFALL_MM
        )

        results["report"] = report_path

        print_check("HTML report generated", report_path is not None)
        if report_path:
            report_size_kb = Path(report_path).stat().st_size / 1024
            print_check(f"Report saved: {Path(report_path).name}", Path(report_path).exists())
            print_check(f"Report size: {report_size_kb:.1f} KB", report_size_kb > 5)

    except Exception as e:
        print(f"  ✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        results["report"] = None

    # ── Final Validation Checklist ────────────────────────────────────────
    elapsed = time.time() - start_time
    print_section("VALIDATION CHECKLIST")

    checks = [
        ("Flood area detected (>0.01 km²)", results["flood_extent"].get("total_area_km2", 0) > 0.01),
        ("Risk zones generated (≥3 total)", len(results["risk_zones"].get("geojson", {}).get("features", [])) >= 3),
        ("≥1 HIGH risk zone", sum(
            1 for f in results["risk_zones"].get("geojson", {}).get("features", [])
            if f["properties"].get("risk_level") == "HIGH"
        ) >= 1),
        ("≥2 MEDIUM risk zones", sum(
            1 for f in results["risk_zones"].get("geojson", {}).get("features", [])
            if f["properties"].get("risk_level") == "MEDIUM"
        ) >= 2),
        ("Impact zones calculated (≥3)", results["impact_zones"].get("feature_count", 0) >= 3),
        ("District statistics computed", results["districts"].get("feature_count", 0) >= 5),
        ("Districts with flooding", results["districts"].get("district_count_affected", 0) >= 1),
        ("Map generated without errors", results.get("map", {}).get("map_path") is not None),
        ("All 4 map layers present", all(results.get("map", {}).get("layers", {}).values())),
        ("Report HTML generated", results.get("report") is not None),
        ("Flood extent GeoJSON saved", results["flood_extent"].get("path") is not None),
        ("Risk zones GeoJSON saved", results["risk_zones"].get("path") is not None),
        ("Impact zones GeoJSON saved", results["impact_zones"].get("path") is not None),
        ("District stats GeoJSON saved", results["districts"].get("path") is not None),
        (f"Execution time <120s ({elapsed:.1f}s)", elapsed < 120),
    ]

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    for label, ok in checks:
        print_check(label, ok)

    print(f"\n{'═' * 60}")
    print(f"  RESULT: {passed}/{total} checks passed")
    print(f"  Execution time: {elapsed:.1f}s")

    if passed == total:
        print("  STATUS: ALL TESTS PASSED")
    elif passed >= total * 0.8:
        print(f"  STATUS: MOSTLY PASSED ({total - passed} minor issues)")
    else:
        print(f"  STATUS: {total - passed} checks failed — review above")

    print(f"{'═' * 60}")

    # Print output file summary
    print("\n  OUTPUT FILES:")
    output_paths = {
        "Interactive Map (HTML)": results.get("map", {}).get("map_path"),
        "Satellite Report (HTML)": results.get("report"),
        "Flood Extent GeoJSON": results["flood_extent"].get("path"),
        "Risk Zones GeoJSON": results["risk_zones"].get("path"),
        "Impact Zones GeoJSON": results["impact_zones"].get("path"),
        "District Stats GeoJSON": results["districts"].get("path"),
    }

    for label, path in output_paths.items():
        if path and Path(path).exists():
            size_kb = Path(path).stat().st_size / 1024
            print(f"  {'✓'} {label}")
            print(f"      {path} ({size_kb:.1f} KB)")
        else:
            print(f"  {'✗'} {label}: not found")

    print()
    return results


if __name__ == "__main__":
    run_analysis()
