try:
    import ee
except ImportError:
    ee = None

# SCL classes to drop: 3=cloud shadow, 8=cloud (medium prob), 9=cloud (high prob),
# 10=thin cirrus, 11=snow/ice. Everything else (incl. vegetation/bare soil/water) is kept.
SCL_MASKED_CLASSES = [3, 8, 9, 10, 11]


def mask_s2_clouds(image):
    """INPUTS: ee.Image, a Sentinel-2 SR scene with an SCL (Scene Classification Layer) band.
    OUTPUTS: same image with cloud/cloud-shadow/cirrus/snow pixels masked out, using SCL
    classes 3, 8, 9, 10, 11. remap() sends those classes to 0 (masked) and leaves every other
    class at 1 (kept) via its default value."""
    scl = image.select("SCL")
    keep_mask = scl.remap(SCL_MASKED_CLASSES, [0] * len(SCL_MASKED_CLASSES), 1).rename("scl_mask")
    return image.updateMask(keep_mask)


def add_ndvi(image):
    """INPUTS: ee.Image with NIR (B8) and RED (B4) bands.
    OUTPUTS: same image with an added 'NDVI' band: (B8-B4)/(B8+B4), vegetation health/cover."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def add_bsi(image):
    """INPUTS: ee.Image with SWIR1 (B11), RED (B4), NIR (B8), BLUE (B2) bands.
    OUTPUTS: same image with an added 'BSI' band (Bare Soil Index - spikes on exposed
    soil from mining scars or clear-cuts): ((B11+B4)-(B8+B2)) / ((B11+B4)+(B8+B2))."""
    bsi = image.expression(
        "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {
            "SWIR1": image.select("B11"),
            "RED": image.select("B4"),
            "NIR": image.select("B8"),
            "BLUE": image.select("B2"),
        },
    ).rename("BSI")
    return image.addBands(bsi)


def add_indices(image):
    """Add NDVI and BSI bands.
    NDVI = normalizedDifference(['B8','B4'])
    BSI = ((SWIR+RED)-(NIR+BLUE)) / ((SWIR+RED)+(NIR+BLUE))
          using B11, B4, B8, B2
    INPUTS: cloud-masked S2 image with those bands
    OUTPUTS: same image with 'ndvi' and 'bsi' added"""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    bsi = image.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {
            "SWIR": image.select("B11"),
            "RED": image.select("B4"),
            "NIR": image.select("B8"),
            "BLUE": image.select("B2"),
        },
    ).rename("bsi")
    return image.addBands([ndvi, bsi])


def prepare_s2(image):
    """INPUTS: raw ee.Image from COPERNICUS/S2_SR_HARMONIZED. OUTPUTS: cloud-masked image with
    NDVI and BSI bands added - the shared prep step used by both baseline and detection passes
    so the two are computed identically."""
    return add_bsi(add_ndvi(mask_s2_clouds(image)))
