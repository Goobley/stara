import astropy.units as u
import numpy as np
from skimage.measure import regionprops, label, regionprops_table
from skimage.filters import median
from skimage.morphology import disk, square
from skimage.util import invert
import sunpy.map
from astropy.table import QTable
from astropy.time import Time
from numba import njit

def erode(im, selem):
    '''
    Morphologically erode image with structure element.
    '''
    padY, padX = [s % 2 == 0 for s in selem.shape]
    padSelem = np.zeros((selem.shape[0]+int(padY), selem.shape[1]+int(padX)))
    padSelem[padY:, padX:] = selem
    padWidth = padSelem.shape[0] // 2
    padIm = np.pad(im, padWidth, mode='reflect')
    return erode_core(im, padIm, padSelem)

@njit
def erode_core(im, padIm, selem):
    padWidth = selem.shape[0] // 2
    out = np.zeros_like(im)
    if padWidth % 2 == 0:
        padWidth -= 1
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            xc = x + padWidth
            yc = y + padWidth
            minVal = padIm[yc,xc]
            for yy in range(selem.shape[0]):
                for xx in range(selem.shape[1]):
                    if selem[yy, xx]:
                        xxc = xx - padWidth
                        yyc = yy - padWidth
                        minVal = min(minVal, padIm[yc + yyc, xc + xxc])
            out[y, x] = minVal
    return out

def dilate(im, selem):
    '''
    Morphologically dilate image with structure element.
    '''
    padY, padX = [s % 2 == 0 for s in selem.shape]
    padSelem = np.zeros((selem.shape[0]+int(padY), selem.shape[1]+int(padX)))
    padSelem[padY:, padX:] = selem
    padWidth = padSelem.shape[0] // 2
    padIm = np.pad(im, padWidth, mode='reflect')
    return dilate_core(im, padIm, padSelem)

@njit
def dilate_core(im, padIm, selem):
    padWidth = selem.shape[0] // 2
    out = np.zeros_like(im)
    if padWidth % 2 == 0:
        padWidth -= 1
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            xc = x + padWidth
            yc = y + padWidth
            maxVal = padIm[yc,xc]
            for yy in range(selem.shape[0]):
                for xx in range(selem.shape[1]):
                    if selem[yy, xx]:
                        xxc = xx - padWidth
                        yyc = yy - padWidth
                        maxVal = max(maxVal, padIm[yc + yyc, xc + xxc])
            out[y, x] = maxVal
    return out

def opening(im, selem):
    '''
    Morphological opening.
    
    Parameters
    ----------
    im : np.ndarray
        The image to be filtered.
    selem : np.ndarray
        The structure element to be used in the morphological opening.
        
    Returns
    -------
    result : np.ndarray
        The morphological opening of the image by the structure element.
    '''
    return dilate(erode(im, selem), selem)

def white_tophat(im : np.ndarray, selem : np.ndarray):
    '''
    Morphological white tophat filter.
    
    Parameters
    ----------
    im : np.ndarray
        The image to be filtered.
    selem : np.ndarray
        The structure element to be used in the morphological opening.
        
    Returns
    -------
    result : np.ndarray
        The white tophat filtered image, i.e. im - opening(im, selem).
    '''
    return im - opening(im, selem)

@u.quantity_input
def stara(
    smap,
    circle_radius: u.deg = 100 * u.arcsec,
    median_box: u.deg = 10 * u.arcsec,
    threshold=6000,
    limb_filter: u.percent = None,
):
    """
    A method for automatically detecting sunspots in white-light data using morphological operations
    Parameters
    ----------
    smap : `sunpy.map.GenericMap`
        The map to apply the algorithm to.
    circle_radius : `astropy.units.Quantity`, optional
        The angular size of the structuring element used in the
        `skimage.morphology.white_tophat`. This is the maximum radius of
        detected features.
    median_box : `astropy.units.Quantity`, optional
        The size of the structuring element for the median filter, features
        smaller than this will be averaged out.
    threshold : `int`, optional
        The threshold used for detection, this will be subject to detector
        degradation. The default is a reasonable value for HMI continuum images.
    limb_filter : `astropy.units.Quantity`, optional
        If set, ignore features close to the limb within a percentage of the
        radius of the disk. A value of 10% generally filters out false
        detections around the limb with HMI continuum images.
    """
    data = invert(smap.data)

    # Filter things that are close to limb to reduce false detections
    if limb_filter is not None:
        hpc_coords = sunpy.map.all_coordinates_from_map(smap)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / (
            smap.rsun_obs - smap.rsun_obs * limb_filter
        )
        data[r > 1] = np.nan

    # Median filter to remove detections based on hot pixels
    m_pix = int((median_box / smap.scale[0]).to_value(u.pix))
    med = median(data, square(m_pix), behavior="ndimage")

    # Construct the pixel structuring element
    c_pix = int(np.round((circle_radius / smap.scale[0]).to_value(u.pix)))
    circle = disk(c_pix // 2)

    finite = white_tophat(med, circle)
    finite[np.isnan(finite)] = 0  # Filter out nans

    return finite > threshold


def get_regions(segmentation, smap, properties=("label", "centroid", "area", "min_intensity")):
    labelled = label(segmentation)
    if labelled.max() == 0:
        return QTable()

    regions = regionprops_table(
        labelled, smap.data, properties=properties
    )

    regions["obstime"] = Time([smap.date] * regions["label"].size)
    regions["center_coord"] = smap.pixel_to_world(
        regions["centroid-0"] * u.pix, regions["centroid-1"] * u.pix
    ).heliographic_stonyhurst

    return QTable(regions)
