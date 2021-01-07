# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


# General purpose libraries
import numpy as np
import os
import sys

# Read annotations
from read_roi import read_roi_zip  # https://github.com/hadim/read-roi
import json

# Create masks
from PIL import Image, ImageDraw
from skimage import draw as skimage_draw
from skimage import morphology
from skimage import measure
from scipy import ndimage

from skimage.io import imsave
import warnings


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

__version__ = "0.0.0"
__author__ = "Florian MUELLER"
__email___ = "muellerf.research@gmail.com"


# ---------------------------------------------------------------------------
# Some helper functions
# ---------------------------------------------------------------------------


def create_folder(folder_new):
    """Function takes as an input a path-name and creates folder if
    the folder does not exist.
    """
    if not os.path.isdir(folder_new):
        os.makedirs(folder_new)


def print_progress(iteration, total, prefix="", suffix="", decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)

    https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    more info: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "*" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write("\r%s |%s| %s%s %s\r" % (prefix, bar, percents, "%", suffix)),

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Classes to import annotations
# ---------------------------------------------------------------------------


class AnnotationImporter:
    """Base class to import manual annoations importer"""

    def load(self, path_open):
        """ Load and annotations and return dictionary with annotations."""

        raise NotImplementedError("No load function defined for this class!")


class GeojsonImporter(AnnotationImporter):
    """ Class to import manual annotations from GeoJson files. """

    def __init__(self, image_size=(2048, 2048)):
        """
        Initiate annotation dictionary.

        Args:
            image_size (tuple): size of image.

        """
        self.image_size = image_size

    def load(self, file_open):
        """
        Read folder content based on defined config.

        Args:
            file_open (string): file-name of annotation.

        Returns:
            annot_dict (dictionary): contains all annotated elements
            roi_size_all (list): contains size of each annotated element
        """

        with open(file_open, encoding="utf-8-sig") as fh:
            data_json = json.load(fh)

        # Overwrite default file size if bounding box is present
        if "bbox" in data_json:
            self.image_size = (
                int(data_json["bbox"][2] - data_json["bbox"][0] + 1),
                int(data_json["bbox"][3] - data_json["bbox"][1] + 1),
            )

        # Loop over list and create simple dictionary & get size of annotations
        annot_dict = {}
        roi_size_all = {}

        skipped = []

        for feat_idx, feat in enumerate(data_json["features"]):

            if feat["geometry"]["type"] not in ["Polygon", "LineString"]:
                skipped.append(feat["geometry"]["type"])
                continue

            # skip empty roi
            if len(feat["geometry"]["coordinates"][0]) <= 0:
                continue

            key_annot = "annot_" + str(feat_idx)
            annot_dict[key_annot] = {}
            annot_dict[key_annot]["type"] = feat["geometry"]["type"]
            annot_dict[key_annot]["pos"] = np.squeeze(
                np.asarray(feat["geometry"]["coordinates"])
            )
            annot_dict[key_annot]["properties"] = feat["properties"]

            # Store size of regions
            if not (feat["properties"]["label"] in roi_size_all):
                roi_size_all[feat["properties"]["label"]] = []

            roi_size_all[feat["properties"]["label"]].append(
                [
                    annot_dict[key_annot]["pos"][:, 0].max()
                    - annot_dict[key_annot]["pos"][:, 0].min(),
                    annot_dict[key_annot]["pos"][:, 1].max()
                    - annot_dict[key_annot]["pos"][:, 1].min(),
                ]
            )

        print("Skipped geometry type(s):", skipped)
        return annot_dict, roi_size_all, self.image_size


# ---------------------------------------------------------------------------
# Classes to generate masks
# ---------------------------------------------------------------------------


class MaskGenerator:
    """Base class for mask generators."""

    def __init__(self):
        pass

    def generate(self, annotDic):
        """  Generate the masks and return a dictionary."""
        raise NotImplementedError("No load function defined for this class!")

    def plot(self):
        """  Plot masks."""
        pass

    def save(self, mask_dict, mask_key, file_name):
        """
        Save selected mask to a png file.

        Args:
            mask_dict (dictionary): dictionary with masks.
            mask_key (string): key for mask that should be saved.
            file_name (string): file-name for mask
        """

        if not (mask_key in mask_dict.keys()):
            print(f"Selected key ({mask_key})is not present in mask dictionary.")
            return

        # Save label - different labels are saved differently
        mask_save = mask_dict[mask_key]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if mask_key is "distance_map":
                imsave(file_name, mask_save)

            elif (mask_key is "edge") or (mask_key is "fill"):
                imsave(file_name, 255 * mask_save)

            elif mask_key is "edge_weighted":
                mask_rescale = (
                    (mask_save - mask_save.min())
                    * 255
                    / (mask_save.max() - mask_save.min())
                )
                mask_rescale = mask_rescale.astype("uint8")
                imsave(file_name, mask_rescale)

            else:
                imsave(file_name, mask_save.astype("float32"))


class BinaryMaskGenerator(MaskGenerator):
    """
    Create binary masks from dictionary with annnotations. Depending on the
    annotation type, different masks are created. If masks are
        >>  polygons  : edge mask and a filled mask are created.
        >> freelines  : only an edge mask is created.


    """

    def __init__(
        self, image_size=(2048, 2048), erose_size=5, obj_size_rem=500, save_indiv=False
    ):
        self.erose_size = erose_size
        self.obj_size_rem = obj_size_rem
        self.save_indiv = save_indiv
        self.image_size = (image_size[1], image_size[0])

    def generate(self, annot_dict):
        """
        Create masks from annotation dictionary

        Args:
            annot_dict (dictionary): dictionary with annotations

        Returns:
            mask_dict (dictionary): dictionary with masks
        """

        # Get dimensions of image and created masks of same size
        # This we need to save somewhere (e.g. as part of the geojson file?)

        # Filled masks and edge mask for polygons
        mask_fill = np.zeros(self.image_size, dtype=np.uint8)
        mask_edge = np.zeros(self.image_size, dtype=np.uint8)
        mask_labels = np.zeros(self.image_size, dtype=np.uint16)

        rr_all = []
        cc_all = []

        if self.save_indiv is True:
            mask_edge_indiv = np.zeros(
                (self.image_size[0], self.image_size[1], len(annot_dict)), dtype=np.bool
            )
            mask_fill_indiv = np.zeros(
                (self.image_size[0], self.image_size[1], len(annot_dict)), dtype=np.bool
            )

        # Image used to draw lines - for edge mask for freelines
        # im_freeline = Image.new('1', self.image_size, color=0)
        im_freeline = Image.new("1", (self.image_size[1], self.image_size[0]), color=0)
        draw = ImageDraw.Draw(im_freeline)

        # Loop over all roi
        i_roi = 0
        for roi_key, roi in annot_dict.items():
            roi_pos = roi["pos"]

            # Check region type

            # freeline - line
            if roi["type"] == "freeline" or roi["type"] == "LineString":

                # Loop over all pairs of points to draw the line

                for ind in range(roi_pos.shape[0] - 1):
                    line_pos = (
                        roi_pos[ind, 1],
                        roi_pos[ind, 0],
                        roi_pos[ind + 1, 1],
                        roi_pos[ind + 1, 0],
                    )
                    draw.line(line_pos, fill=1, width=self.erose_size)

            # freehand - polygon
            elif (
                roi["type"] == "freehand"
                or roi["type"] == "polygon"
                or roi["type"] == "polyline"
                or roi["type"] == "Polygon"
            ):

                # Draw polygon
                rr, cc = skimage_draw.polygon(
                    [self.image_size[0] - r for r in roi_pos[:, 1]], roi_pos[:, 0]
                )

                # Make sure it's not outside
                rr[rr < 0] = 0
                rr[rr > self.image_size[0] - 1] = self.image_size[0] - 1

                cc[cc < 0] = 0
                cc[cc > self.image_size[1] - 1] = self.image_size[1] - 1

                # Test if this region has already been added
                if any(np.array_equal(rr, rr_test) for rr_test in rr_all) and any(
                    np.array_equal(cc, cc_test) for cc_test in cc_all
                ):
                    # print('Region #{} has already been used'.format(i +
                    # 1))
                    continue

                rr_all.append(rr)
                cc_all.append(cc)

                # Generate mask
                mask_fill_roi = np.zeros(self.image_size, dtype=np.uint8)
                mask_fill_roi[rr, cc] = 1

                # Erode to get cell edge - both arrays are boolean to be used as
                # index arrays later
                mask_fill_roi_erode = morphology.binary_erosion(
                    mask_fill_roi, np.ones((self.erose_size, self.erose_size))
                )
                mask_edge_roi = (
                    mask_fill_roi.astype("int") - mask_fill_roi_erode.astype("int")
                ).astype("bool")

                # Save array for mask and edge
                mask_fill[mask_fill_roi > 0] = 1
                mask_edge[mask_edge_roi] = 1
                mask_labels[mask_fill_roi > 0] = i_roi + 1

                if self.save_indiv is True:
                    mask_edge_indiv[:, :, i_roi] = mask_edge_roi.astype("bool")
                    mask_fill_indiv[:, :, i_roi] = mask_fill_roi_erode.astype("bool")

                i_roi = i_roi + 1

            else:
                roi_type = roi["type"]
                raise NotImplementedError(
                    f'Mask for roi type "{roi_type}" can not be created'
                )

        del draw

        # Convert mask from free-lines to numpy array
        mask_edge_freeline = np.asarray(im_freeline)
        mask_edge_freeline = mask_edge_freeline.astype("bool")

        # Post-processing of fill and edge mask - if defined
        mask_dict = {}
        if np.any(mask_fill):

            # (1) remove edges , (2) remove small  objects
            mask_fill = mask_fill & ~mask_edge
            mask_fill = morphology.remove_small_objects(
                mask_fill.astype("bool"), self.obj_size_rem
            )

            # For edge - consider also freeline edge mask

            mask_edge = mask_edge.astype("bool")
            mask_edge = np.logical_or(mask_edge, mask_edge_freeline)

            # Assign to dictionary for return
            mask_dict["edge"] = mask_edge
            mask_dict["fill"] = mask_fill.astype("bool")
            mask_dict["labels"] = mask_labels.astype("uint16")

            if self.save_indiv is True:
                mask_dict["edge_indiv"] = mask_edge_indiv
                mask_dict["fill_indiv"] = mask_fill_indiv
            else:
                mask_dict["edge_indiv"] = np.zeros(
                    self.image_size + (1,), dtype=np.uint8
                )
                mask_dict["fill_indiv"] = np.zeros(
                    self.image_size + (1,), dtype=np.uint8
                )

        # Only edge mask present
        elif np.any(mask_edge_freeline):
            mask_dict["edge"] = mask_edge_freeline
            mask_dict["fill"] = mask_fill.astype("bool")
            mask_dict["labels"] = mask_labels.astype("uint16")

            mask_dict["edge_indiv"] = np.zeros(self.image_size + (1,), dtype=np.uint8)
            mask_dict["fill_indiv"] = np.zeros(self.image_size + (1,), dtype=np.uint8)

        else:
            raise Exception("No mask has been created.")

        return mask_dict


class DistanceMapGenerator(MaskGenerator):
    """
    Create a distance transform from the edge. Stored as 16bit float, for
    display and saving converted to float32 (.astype('float32'))

    Requires that binary weights are calculated first, which is generated with
    the BinaryMaskGenerator (with the option flag_save_indiv=True).


    """

    def __init__(self, truncate_distance=None):
        self.truncate_distance = truncate_distance

    def generate(self, annot_dict, mask_dict):
        """
        Creates a distance map with truncated distance to the edge of the cell.

        Args:
            annot_dict (dictionary): dictionary with annotations
            mask_dict (dictionary): dictionary with masks containing at
                                    least binary masks

        Returns:
            mask_dict (dictionary): dictionary with additional weighted masks
        """

        mask_fill_indiv = mask_dict["fill_indiv"]
        mask_edge_indiv = mask_dict["edge_indiv"]
        dist_mat = np.ones(np.shape(mask_fill_indiv))

        for i_cell in range(mask_fill_indiv.shape[-1]):
            img_cell = mask_edge_indiv[:, :, i_cell] + mask_fill_indiv[:, :, i_cell]

            dist_cell = ndimage.distance_transform_edt(img_cell)
            if self.truncate_distance:
                dist_cell[dist_cell > self.truncate_distance] = self.truncate_distance
            dist_mat[:, :, i_cell] = dist_cell

        dist_map = np.sum(dist_mat, 2)

        # Note: saved as uint 16
        mask_dict["distance_map"] = dist_map.astype("uint16")

        return mask_dict


class WeightedEdgeMaskGenerator(MaskGenerator):
    """
    Create a weighted edge mask that depend on distance to two closests cells.
    Reference: https://arxiv.org/abs/1505.04597

    Requires that binary weights are calculated first, which is generated with
    the BinaryMaskGenerator (with the option flag_save_indiv=True).

    Results are saved in a dictionary with the key 'mask_edge_weighted'
    """

    def __init__(self, sigma=8, w0=10):
        self.sigma = sigma
        self.w0 = w0

    def generate(self, annot_dict, mask_dict):
        """
        Create masks.

        Args:
            annot_dict (dictionary): dictionary with annotations
            mask_dict (dictionary): dictionary with masks containing at
                                    least binary masks

        Returns:
            mask_dict (dictionary): dictionary with additional weighted masks
        """

        mask_fill = mask_dict["fill"]
        mask_edge_indiv = mask_dict["edge_indiv"]

        # Calculating the weigth w that balance the pixel frequency
        x = (mask_fill > 0).astype("int")

        # Percentage of image being a cell
        ratio = float(x.sum()) / float(x.size - x.sum())

        if ratio == 0:
            mask_dict["edge_weighted"] = None
            return

        if ratio < 1.0:
            wc = (1 / ratio, 1)
        else:
            wc = (1, 1 / ratio)

        # Calculate the distance map from each pixel to every cell
        dist_mat = np.ones(np.shape(mask_edge_indiv))
        image_ones = np.ones(np.shape(mask_fill))

        for i_cell in range(mask_edge_indiv.shape[-1]):

            edge_cell_inverted = image_ones - 1 * mask_edge_indiv[:, :, i_cell]
            dist_mat[:, :, i_cell] = ndimage.distance_transform_edt(edge_cell_inverted)

        # Sort distance map and use only the two closest cells and add them
        # up
        dist_map = np.sum(np.sort(dist_mat)[:, :, (0, 1)], 2)

        # Calculated exponential weight for each pixel
        exp_weigth = self.w0 * np.exp(-((dist_map) ** 2) / (2 * self.sigma ** 2))

        # Calculate frequency weight
        wc_map = mask_fill * wc[0] + (1 - mask_fill) * wc[1]
        mask_edge = wc_map + exp_weigth

        # Sum of both weights
        # Note: saved as float 16 - to plot has to be converted to float32
        # To be saved rescaled as 8 bit
        mask_dict["edge_weighted"] = mask_edge.astype("float16")

        return mask_dict


class BorderMaskGenerator(MaskGenerator):
    """
    https://github.com/selimsef/dsb2018_topcoders
    """

    def __init__(self, border_detection_threshold=6):
        self.border_detection_threshold = border_detection_threshold

    def generate(self, annot_dict, mask_dict):
        labels = mask_dict["labels"]
        tmp = mask_dict["edge"] > 0
        tmp = morphology.dilation(
            tmp, morphology.square(self.border_detection_threshold)
        )

        props = measure.regionprops(labels)
        msk0 = 255 * (labels > 0)
        msk0 = msk0.astype("uint8")

        msk1 = np.zeros_like(labels, dtype="bool")

        max_area = np.max([p.area for p in props])

        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                sz = self.border_detection_threshold

                uniq = np.unique(
                    labels[
                        max(0, y0 - sz) : min(labels.shape[0], y0 + sz + 1),
                        max(0, x0 - sz) : min(labels.shape[1], x0 + sz + 1),
                    ]
                )
                if len(uniq[uniq > 0]) > 1:
                    msk1[y0, x0] = True
                    msk0[y0, x0] = 0

        msk0 = 255 * (labels > 0)
        msk0 = msk0.astype("uint8")  # cell area
        msk1 = morphology.binary_closing(msk1)
        msk1 = 255 * msk1  # cell boundarys
        msk1 = msk1.astype("uint8")

        msk2 = np.zeros_like(labels, dtype="uint8")
        msk = np.stack((msk0, msk1, msk2))
        msk = np.rollaxis(msk, 0, 3)

        # Note: saved as float 16 - to plot has to be converted to float32
        # To be saved rescaled as 8 bit
        mask_dict["border_mask"] = msk.astype("float32")

        return mask_dict
