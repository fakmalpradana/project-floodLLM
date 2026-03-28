import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter

def detect_water_sar(input_path: str, output_path: str = None) -> tuple[np.ndarray, dict]:
    """
    Detects water from Sentinel-1 GRD imagery using Otsu's dynamic thresholding.
    Assumes input is a pre-processed SAR intensity or backscatter image (VV or VH).
    
    Args:
        input_path (str): Path to the input SAR GeoTIFF.
        output_path (str, optional): Path to save the binary water mask.
        
    Returns:
        tuple: (binary_water_mask, profile)
               - binary_water_mask: 2D numpy array (1=water, 0=non-water)
               - profile: rasterio profile dictionary with CRS and transform
    """
    with rasterio.open(input_path) as src:
        sar_data = src.read(1)
        profile = src.profile

    # Handle nodata/nan values
    valid_mask = ~np.isnan(sar_data) & (sar_data != src.nodata)
    valid_data = sar_data[valid_mask]

    if valid_data.size == 0:
        raise ValueError("No valid data found in SAR image.")

    # Convert to dB if not already (assuming linear backscatter > 0)
    # This is a heuristic check: if max value is low, it might already be dB, but typically linear targets are > 0.
    if np.nanmax(valid_data) < 100 and np.nanmin(valid_data) >= 0:
        valid_data_db = 10 * np.log10(np.clip(valid_data, 1e-6, None))
        sar_data_db = np.full_like(sar_data, np.nan, dtype=np.float32)
        sar_data_db[valid_mask] = valid_data_db
    else:
        sar_data_db = sar_data
        valid_data_db = valid_data

    # Apply spatial filtering (median filter) to reduce speckle noise
    filtered_data = median_filter(sar_data_db, size=3)
    filtered_valid_data = filtered_data[valid_mask]

    # Calculate Otsu threshold dynamically
    try:
        otsu_val = threshold_otsu(filtered_valid_data)
    except Exception as e:
        # Fallback to a generalized threshold for SAR water (e.g., -15 dB)
        otsu_val = -15.0

    # Water is typically characterized by low backscatter in SAR (dark pixels)
    water_mask = np.zeros_like(sar_data, dtype=np.uint8)
    water_mask[valid_mask & (filtered_data < otsu_val)] = 1

    # Update profile for output mask
    profile.update({
        'dtype': 'uint8',
        'count': 1,
        'nodata': 255
    })
    
    # Keep nodata pixels marked optionally (e.g., as 255)
    water_mask[~valid_mask] = 255

    # Write output if a path is provided
    if output_path:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(water_mask, 1)

    return water_mask, profile
