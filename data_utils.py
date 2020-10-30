import numpy as np
from geojson import Polygon as geojson_polygon
from shapely.geometry import Polygon as shapely_polygon
from geojson import Feature, FeatureCollection, dump
from skimage import measure, morphology
import matplotlib.pyplot as plt

def mask_to_geojson(img_mask, label=None, simplify_tol=1.5):
    """
    Args:
      img_mask (numpy array): numpy data, with each object being assigned with a unique uint number
      label (str): like 'cell', 'nuclei'
      simplify_tol (float): give a higher number if you want less coordinates.
    """
    # for img_mask, for cells on border, should make sure on border pixels are # set to 0
    shape_x, shape_y = img_mask.shape
    shape_x, shape_y = shape_x - 1, shape_y - 1
    img_mask[0, :] = img_mask[:, 0] = img_mask[shape_x, :] = img_mask[:, shape_y] = 0
    features = []
    label = label or "cell"
    # Get all object ids, remove 0 since this is background
    ind_objs = np.unique(img_mask)
    ind_objs = np.delete(ind_objs, np.where(ind_objs == 0))
    for obj_int in np.nditer(ind_objs, flags=["zerosize_ok"]):
        # Create binary mask for current object and find contour
        img_mask_loop = np.zeros((img_mask.shape[0], img_mask.shape[1]))
        img_mask_loop[img_mask == obj_int] = 1
        contours_find = measure.find_contours(img_mask_loop, 0.5)
        if len(contours_find) == 1:
            index = 0
        else:
            pixels = []
            for _, item in enumerate(contours_find):
                pixels.append(len(item))
            index = np.argmax(pixels)
        contour = contours_find[index]

        contour_as_numpy = contour[:, np.argsort([1, 0])]
        contour_as_numpy[:, 1] = np.array([img_mask.shape[0] - h[0] for h in contour])
        contour_asList = contour_as_numpy.tolist()

        if simplify_tol is not None:
            poly_shapely = shapely_polygon(contour_asList)
            poly_shapely_simple = poly_shapely.simplify(
                simplify_tol, preserve_topology=False
            )
            contour_asList = list(poly_shapely_simple.exterior.coords)
            contour_as_Numpy = np.asarray(contour_asList)

        # Create and append feature for geojson
        pol_loop = geojson_polygon([contour_asList])

        full_label = label + "_idx"
        index_number = int(obj_int - 1)
        features.append(
            Feature(
                geometry=pol_loop, properties={full_label: index_number, "label": label}
            )
        )

    # feature_collection = FeatureCollection(
    #    features, bbox=[0, 0, img_mask.shape[1] - 1, img_mask.shape[0] - 1]
    # )
    features = list(
        map(
            lambda feature: np.array(
                feature["geometry"]["coordinates"][0], dtype="uint16"
            ),
            features,
        )
    )
    return features  # feature_collection

def visualize(images, masks, original_image=None, original_mask=None):
    fontsize = 18
    assert len(images) == len(masks)
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, len(images), figsize=(25, 10))
        for i in range(len(images)):
          ax[0,i].imshow(images[i])
          ax[1,i].imshow(masks[i])
        plt.savefig('./data/augmented_grid.png')
    else:
        f, ax = plt.subplots(2, len(images)+1, figsize=(25,10))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
          
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        for i in range(len(images)):  
          ax[0, i+1].imshow(images[i])
          ax[0, i+1].set_title('Augmented image', fontsize=fontsize)
          
          ax[1, i+1].imshow(masks[i])
          ax[1, i+1].set_title('Augmented mask', fontsize=fontsize)
        plt.savefig('./data/augmented_grid.png')
    return f