import os
import numpy as np
from geojson import Polygon as geojson_polygon
from shapely.geometry import Polygon as shapely_polygon
from geojson import Feature, FeatureCollection, dump
import io
import json


def read_from_json(json_file_path):
    """Function to read json file (annotation file)
    """
    with io.open(json_file_path, "r", encoding="utf-8-sig") as myfile:
        data = json.load(myfile)
    return data


def write_to_json(data, json_file_path):
    """Function to write (updated) annotation to json file
    """
    with io.open(json_file_path, "w", encoding="utf-8-sig") as myfile:
        json.dump(data, myfile)
        myfile.close()


def simplify_contour(contour_asList, simplify_tol=0.2):
    poly_shapely = shapely_polygon(contour_asList)
    poly_shapely_simple = poly_shapely.simplify(
        simplify_tol, preserve_topology=True
    )
    contour_asList = list(poly_shapely_simple.exterior.coords)
    return contour_asList


def simplify_annotations(json_file_path, simplify_tol=0.2):
    json_data = read_from_json(json_file_path)
    for i, feature in enumerate(json_data['features']):
        contour_asList = feature['geometry']['coordinates'][0]
        contour_asList = simplify_contour(contour_asList, simplify_tol)
        json_data['features'][i]['geometry']['coordinates'][0] = contour_asList
    return json_data


def rewrite_simplified_json(json_file_path):
    json_data = simplify_annotations(json_file_path, 0.2)
    write_to_json(json_data, json_file_path)
    #gen_mask_from_geojson(['./test/test.json'], masks_to_create_value='border_mask')
    #gen_mask_from_geojson(['./annotation.json'], masks_to_create_value='border_mask')


if __name__ == '__main__':
    data_dir = '/Users/hao.xu/Downloads/hpa_dataset_v2'
    for root, _, files in os.walk(data_dir):
        if 'annotation.json' in files:
            json_file_path = os.path.join(root, 'annotation.json')
            rewrite_simplified_json(json_file_path)