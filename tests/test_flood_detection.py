"""Test flood detection processing."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sar_threshold():
    """Test SAR water threshold calculation."""
    print("\n=== Testing SAR Threshold ===")

    from app.processing.sar_processor import SARProcessor

    processor = SARProcessor()

    # Create synthetic VV backscatter data
    # Water typically has VV < -17 dB, land > -17 dB
    vv_data = np.random.normal(-15, 8, (100, 100)).astype(np.float32)

    # Add some "water" pixels
    water_area = np.random.choice([0, 1], (100, 100), p=[0.8, 0.2])
    vv_data[water_area == 1] = np.random.normal(-22, 3, np.sum(water_area == 1))

    # Test Otsu threshold
    try:
        threshold = processor._calculate_otsu_threshold(vv_data)
        print(f"Otsu threshold: {threshold:.2f} dB")

        # Threshold should be between water and land
        if -25 < threshold < -10:
            print("✓ Threshold in reasonable range")
            return True
        else:
            print(f"✗ Threshold out of range: {threshold}")
            return False

    except Exception as e:
        print(f"✗ Threshold calculation failed: {e}")
        return False


def test_mask_postprocessing():
    """Test mask cleaning operations."""
    print("\n=== Testing Mask Post-processing ===")

    from app.processing.sar_processor import SARProcessor

    processor = SARProcessor()

    # Create noisy mask
    mask = np.random.random((50, 50)) > 0.7

    # Add small noise
    mask[5:7, 5:7] = True
    mask[40:41, 40:41] = True

    cleaned = processor._post_process_mask(mask)

    print(f"Original water pixels: {np.sum(mask)}")
    print(f"Cleaned water pixels: {np.sum(cleaned)}")

    # Small objects should be removed
    if np.sum(cleaned) <= np.sum(mask):
        print("✓ Post-processing completed")
        return True

    return False


def test_flood_statistics():
    """Test flood area calculation."""
    print("\n=== Testing Flood Statistics ===")

    from app.processing.sar_processor import SARProcessor
    from affine import Affine

    processor = SARProcessor()

    # Create test mask (20% flooded)
    mask = np.zeros((100, 100), dtype=bool)
    mask[:20, :] = True  # 20% flooded

    # Create transform (10m resolution)
    transform = Affine(10, 0, 0, 0, -10, 0)

    stats = processor._calculate_flood_stats(mask, transform, (0, 0, 1, 1))

    print(f"Flood area: {stats['flood_area_km2']} km²")
    print(f"Flood percentage: {stats['flood_percentage']}%")

    # 20% of 1km x 1km = 0.2 km²
    expected_area = 0.2
    if abs(stats['flood_area_km2'] - expected_area) < 0.01:
        print("✓ Area calculation correct")
        return True
    else:
        print(f"✗ Area mismatch. Expected: {expected_area}, Got: {stats['flood_area_km2']}")
        return False


def test_optical_ndwi():
    """Test NDWI calculation."""
    print("\n=== Testing NDWI Calculation ===")

    from app.processing.optical import OpticalProcessor

    processor = OpticalProcessor()

    # Create synthetic bands
    # Water: high green reflectance, low NIR
    green_water = np.full((50, 50), 0.3, dtype=np.float32)
    nir_water = np.full((50, 50), 0.1, dtype=np.float32)

    # Land: lower green, higher NIR
    green_land = np.full((50, 50), 0.2, dtype=np.float32)
    nir_land = np.full((50, 50), 0.4, dtype=np.float32)

    # Combine
    green = np.hstack([green_water, green_land])
    nir = np.hstack([nir_water, nir_land])

    ndwi = processor._compute_ndwi(green, nir)

    # Water should have positive NDWI, land negative
    water_ndwi = np.mean(ndwi[:, :50])
    land_ndwi = np.mean(ndwi[:, 50:])

    print(f"Water NDWI: {water_ndwi:.3f}")
    print(f"Land NDWI: {land_ndwi:.3f}")

    if water_ndwi > 0 and land_ndwi < 0:
        print("✓ NDWI correctly distinguishes water")
        return True
    else:
        print("✗ NDWI calculation issue")
        return False


def run_tests():
    """Run all flood detection tests."""
    print("=" * 50)
    print("Flood Detection Test Suite")
    print("=" * 50)

    results = []
    results.append(("SAR Threshold", test_sar_threshold()))
    results.append(("Mask Post-processing", test_mask_postprocessing()))
    results.append(("Flood Statistics", test_flood_statistics()))
    results.append(("NDWI Calculation", test_optical_ndwi()))

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
