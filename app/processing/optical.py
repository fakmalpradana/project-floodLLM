import numpy as np
import rasterio

def calculate_ndwi_and_mask(input_path: str, green_band_idx: int = 1, nir_band_idx: int = 2, 
                            threshold: float = 0.0, output_path: str = None) -> tuple[np.ndarray, dict]:
    """
    Calculates the NDWI from a multispectral Optical image (e.g., Sentinel-2)
    and generates a binary water mask. NDWI = (Green - NIR) / (Green + NIR).
    
    Args:
        input_path (str): Path to the stacked input optical GeoTIFF containing Green and NIR bands.
        green_band_idx (int): 1-based index for the Green band (e.g., S2 Band 3).
        nir_band_idx (int): 1-based index for the NIR band (e.g., S2 Band 8).
        threshold (float): NDWI threshold strictly above which is classified as water. Usually 0.0 or 0.1.
        output_path (str, optional): Path to save the binary water mask GeoTIFF.
        
    Returns:
        tuple: (water_mask, profile)
               - water_mask: 2D numpy array (1=water, 0=non-water)
               - profile: rasterio spatial metadata profile
    """
    with rasterio.open(input_path) as src:
        green = src.read(green_band_idx).astype(np.float32)
        nir = src.read(nir_band_idx).astype(np.float32)
        profile = src.profile
        nodata = src.nodata

    # Create validity mask
    valid_mask = np.ones_like(green, dtype=bool)
    if nodata is not None:
        valid_mask = (green != nodata) & (nir != nodata)
    
    # Avoid division by zero
    denominator = (green + nir)
    denominator_safe = np.where(denominator == 0, 1e-10, denominator)
    
    # Calculate NDWI
    ndwi = np.zeros_like(green, dtype=np.float32)
    ndwi_valid = (green[valid_mask] - nir[valid_mask]) / denominator_safe[valid_mask]
    ndwi[valid_mask] = ndwi_valid

    # Apply threshold to create binary mask
    water_mask = np.zeros_like(ndwi, dtype=np.uint8)
    water_mask[valid_mask & (ndwi > threshold)] = 1
    
    # Set nodata values appropriately
    if nodata is not None:
        water_mask[~valid_mask] = 255
        
    # Update raster profile for writing the mask
    profile.update({
        'dtype': 'uint8',
        'count': 1,
        'nodata': 255
    })
    
    if output_path:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(water_mask, 1)
            
    return water_mask, profile
