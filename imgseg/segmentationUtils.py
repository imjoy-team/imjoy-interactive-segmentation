# IMPORTS
import numpy as np

from skimage import measure
from skimage import morphology
from skimage import filters
from skimage import dtype_limits
from skimage.morphology import disk
from skimage.morphology import greyreconstruct
from skimage.segmentation import find_boundaries
from skimage.filters import threshold_otsu

from scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage
from scipy import signal
from imageio import imwrite

import palettable  # pip install palettable

# IMPORTS to create polygons from segmentation masks

from shapely.geometry import (
    Polygon as shapely_polygon,
)  # Used to simplify the mask polygons
from descartes import (
    PolygonPatch,
)  # To plot polygons as patches (https://bitbucket.org/sgillies/descartes/src): pip install descartes
from geojson import Polygon as geojson_polygon
from geojson import (
    Feature,
    FeatureCollection,
    dump,
)  # Used to create and save the geojson files: pip install geojson


def _add_constant_clip(img, const_value):
    """Add constant to the image while handling overflow issues gracefully."""
    min_dtype, max_dtype = dtype_limits(img, clip_negative=False)

    if const_value > (max_dtype - min_dtype):
        raise ValueError(
            "The added constant is not compatible" "with the image data type."
        )

    result = img + const_value
    result[img > max_dtype - const_value] = max_dtype
    return result


def _subtract_constant_clip(img, const_value):
    """Subtract constant from image while handling underflow issues."""
    min_dtype, max_dtype = dtype_limits(img, clip_negative=False)

    if const_value > (max_dtype - min_dtype):
        raise ValueError(
            "The subtracted constant is not compatible" "with the image data type."
        )

    result = img - const_value
    result[img < (const_value + min_dtype)] = min_dtype
    return result


def extended_minima(img, h, selem=None):

    if np.issubdtype(img.dtype, "half"):
        resolution = 2 * np.finfo(img.dtype).resolution
        if h < resolution:
            h = resolution
        h_corrected = h - resolution / 2.0
        shifted_img = img + h
    else:
        shifted_img = _add_constant_clip(img, h)
        h_corrected = h

    rec_img = greyreconstruct.reconstruction(
        shifted_img, img, method="erosion", selem=selem
    )
    residue_img = rec_img - img
    h_min = np.zeros(img.shape, dtype=np.uint8)
    h_min[residue_img > 0] = 1
    return h_min


def segment_cells_nuclei(
    image_input,
    image_predicted,
    h_threshold=15,
    min_size_cell=200,
    min_size_nuclei=1000,
    save_path=None,
):
    """Segment cells and nuclei.
    ARGS
        image_output ... multichannel image. 1st channel is mask of the cells,
                         2nd channel mask of the nuclei.
       image_input  ... image of the cells used for segmentation.
    """

    im_mask_cell = image_predicted[:, :, 0]
    im_mask_nuc = image_predicted[:, :, 1]

    img_cell = image_input[:, :, 0]

    # Segment the nuclei
    nuclei_mask = segment_nuclei_cellcog(
        im_mask_nuc, h_threshold, min_size=min_size_nuclei
    )

    # Segment the cells
    im_binary_output = im_mask_cell > threshold_otsu(im_mask_cell)
    im_binary_output = binary_fill_holes(im_binary_output)
    im_binary_output = morphology.remove_small_objects(
        im_binary_output, min_size=min_size_cell, connectivity=1, in_place=False
    )

    # Apply watershed
    seg = morphology.watershed(
        1.0 - img_cell / 255.0, nuclei_mask, mask=im_binary_output
    )
    cytoplasm_mask = seg

    if save_path:

        from skimage.color import label2rgb

        imwrite(save_path + "_cells_mask.png", np.float32(cytoplasm_mask))

        seg = measure.label(cytoplasm_mask)
        bound = find_boundaries(seg, background=0)

        image_label_overlay = label2rgb(
            seg,
            bg_label=0,
            bg_color=(0.8, 0.8, 0.8),
            colors=palettable.colorbrewer.sequential.YlGn_9.mpl_colors,
        )
        image_label_overlay[bound == 1, :] = 0

        imwrite(save_path + "_cells_color_mask.png", np.float32(image_label_overlay))

        # tiff.imwrite(img_name,np.float32(_nuclei_mask))
        imwrite(save_path + "_nuclei_mask.png", np.float32(nuclei_mask))

        seg = measure.label(nuclei_mask)
        bound = find_boundaries(seg, background=0)
        image_label_overlay = label2rgb(
            seg,
            bg_label=0,
            bg_color=(0.8, 0.8, 0.8),
            colors=palettable.colorbrewer.sequential.YlGn_9.mpl_colors,
        )
        image_label_overlay[bound == 1, :] = 0
        imwrite(save_path + "_nuclei_color_mask.png", np.float32(image_label_overlay))

    return cytoplasm_mask, nuclei_mask


def segment_nuclei_cellcog(im, h_threshold=15, bg_window_size=100, min_size=1000):
    im = im.astype("double")
    im = (im - im.min()) / im.max() * 255
    im = im.astype("uint8")

    # Pre-processing

    # Median filtering
    im_filt = filters.median(im, selem=disk(10))

    # BGD estimation
    window = np.ones((bg_window_size, bg_window_size)) / (
        bg_window_size * bg_window_size
    )
    a = signal.fftconvolve(im_filt, window)
    crop_d = (bg_window_size - 1) // 2
    bgd_crop = a[crop_d : np.shape(im)[0] + crop_d, crop_d : np.shape(im)[1] + crop_d]

    # BGD substraction and clip
    im_prefilt = im_filt - bgd_crop
    im_prefilt = im_prefilt.clip(min=0)

    # Thresholding, fill and remove small objects
    threshold = filters.threshold_otsu(im)
    img_threshold = im > threshold

    img_threshold = morphology.remove_small_objects(img_threshold, min_size)
    img_threshold = ndimage.morphology.binary_fill_holes(img_threshold)

    # Distance transform
    distance = ndimage.distance_transform_edt(img_threshold)
    distance = filters.gaussian(distance, sigma=1)

    # h-maxima detection
    res = extended_minima(-distance, h_threshold)
    label_nuc = measure.label(res)

    # watershed
    wat = morphology.watershed(-distance, label_nuc)
    result_label_seg = morphology.remove_small_objects(wat * img_threshold, min_size)
    return result_label_seg


def masks_to_polygon(
    img_mask, label=None, simplify_tol=0, plot_simplify=False, save_name=None
):
    """
     Find contours with skimage, simplify them (optional), store as geojson:

     1. Loops over each detected object, creates a mask and finds it contour
     2. Contour can be simplified (reduce the number of points)
         - uses shapely: https://shapely.readthedocs.io/en/stable/manual.html#object.simplify
          - will be performed if tolernace simplify_tol is != 0
    3. Polygons will be saved in geojson format, which can be read by ImJoys'
       AnnotationTool. Annotations for one image are stored as one feature collection
       each annotation is one feature:
          "type": "Feature",
           "geometry": {"type": "Polygon","coordinates": [[]]}
           "properties": null

     Args:
         img_mask (2D numpy array): image wiht segmentation masks. Background is 0,
                                 each object has a unique pixel value.
         simplify_tol (float): tolerance for simplification (All points in the simplified object
                               will be within the tolerance distance of the original geometry)
                               No simplification will be performed when set to 0.
         plot_simplify (Boolean): plot results of simplifcation. Plot will be shown for EACH mask.
                                  Use better for debuggin only.
         save_name (string): full file-name to save GeoJson file. Not file will be
                             saved when None.

     Returns:
         contours (List): contains polygon of each object stored as a numpy array.
         feature_collection : GeoJson feature collection
    """
    # for img_mask, for cells on border, should make sSure on border pixels are # set to 0
    shape_x, shape_y = img_mask.shape
    shape_x, shape_y = shape_x - 1, shape_y - 1
    img_mask[0, :] = img_mask[:, 0] = img_mask[shape_x, :] = img_mask[:, shape_y] = 0
    # Prepare list to store polygon coordinates and geojson features
    features = []
    contours = []

    # Get all object ids, remove 0 since this is background
    ind_objs = np.unique(img_mask)
    ind_objs = np.delete(ind_objs, np.where(ind_objs == 0))

    # Loop over all masks
    for obj_int in np.nditer(ind_objs):

        # Create binary mask for current object and find contour
        img_mask_loop = np.zeros((img_mask.shape[0], img_mask.shape[1]))
        img_mask_loop[img_mask == obj_int] = 1
        contour = measure.find_contours(img_mask_loop, 0.5)

        # Proceeed only if one contour was found
        if len(contour) == 1:

            contour_asNumpy = contour[0][:, np.argsort([1, 0])]
            contour_asNumpy[:, 1] = np.array(
                [img_mask.shape[0] - h[0] for h in contour[0]]
            )
            contour_asList = contour_asNumpy.tolist()

            # Simplify polygon if tolerance is set to any value except 0
            if simplify_tol != 0:
                poly_shapely = shapely_polygon(contour_asList)
                poly_shapely_simple = poly_shapely.simplify(
                    simplify_tol, preserve_topology=False
                )
                contour_asList = list(poly_shapely_simple.exterior.coords)
                contour_asNumpy = np.asarray(contour_asList)

                if plot_simplify:
                    plot_polygons(poly_shapely, poly_shapely_simple, obj_int)

            # Append to polygon list
            contours.append(contour_asNumpy)

            # Create and append feature for geojson
            pol_loop = geojson_polygon([contour_asList])
            features.append(Feature(geometry=pol_loop, properties={"label": label}))

        # elif len(contour) == 0:
        #    print(f'No contour found for object {obj_int}')
        # else:
        #    print(f'More than one contour found for object {obj_int}')

    # Save to json file
    if save_name:
        feature_collection = FeatureCollection(
            features, bbox=[0, 0, img_mask.shape[1] - 1, img_mask.shape[0] - 1]
        )
        with open(save_name, "w") as f:
            dump(feature_collection, f)
            f.close()

    return features, contours


def plot_polygons(
    poly_shapely, poly_shapely_simple, fig_title="Shapely polygon simplification"
):
    """
    Function plots two shapely polygons using the descartes library.

    Args:
        poly_shapely (string): shapely polygon before simplification
        poly_shapely (string): shapely polygon after simplification
        fig_title (string): title of figure window
    """
    import matplotlib.pyplot as plt

    # Create a matplotlib figure
    fig = plt.figure(num=fig_title, figsize=(10, 4), dpi=180)

    # *** Original polygon
    ax = fig.add_subplot(121)

    # Make the polygon into a patch and add it to the subplot
    patch = PolygonPatch(poly_shapely, facecolor="#32c61b", edgecolor="#999999")
    ax.add_patch(patch)

    # Fit the figure around the polygon's bounds, render, and save
    minx, miny, maxx, maxy = poly_shapely.bounds
    w, h = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
    ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
    ax.set_aspect(1)
    ax.set_title("Original mask")

    # Plot
    ax = fig.add_subplot(122)

    # Make the polygon into a patch and add it to the subplot
    patch = PolygonPatch(poly_shapely_simple, facecolor="#32c61b", edgecolor="#999999")
    ax.add_patch(patch)

    # Fit the figure around the polygon's bounds, render, and save
    w, h = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
    ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
    ax.set_aspect(1)
    ax.set_title("Simplified mask")
