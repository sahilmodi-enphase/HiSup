import cv2
import json
import numpy as np
import os
import re

# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def create_roofai_dataset_coco_annotations():
    dataset_folder = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k/train"
    image_folder = os.path.join(dataset_folder, "images")
    contour_json_file = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k/vgg_till_oct31_outer_contour.json"
    ann_out_file = os.path.join(dataset_folder, "annotation.json")

    with open(contour_json_file, "r") as f:
        contour_data = json.load(f)

    dataset = {}
    dataset['categories'] = [{'id': 100, 'name': 'building', 'supercategory': 'building'}]
    images_list = []
    annotations_list = []

    for filename in os.listdir(image_folder):
        outer_contour = contour_data[filename]
        segmentation = [[x for y in outer_contour for x in y]]
        segmentation[0].extend(segmentation[0][0:2])
        min_x = min(outer_contour, key=lambda x: x[0])[0]
        min_y = min(outer_contour, key=lambda x: x[1])[1]
        max_x = max(outer_contour, key=lambda x: x[0])[0]
        max_y = max(outer_contour, key=lambda x: x[1])[1]
        images_list.append({'file_name': filename,
                            'height': 800,
                            'width': 800,
                            'id': int(re.findall("\d+", filename)[0])})
        annotations_list.append({
            "area": PolyArea([x for x, y in outer_contour], [y for x, y in outer_contour]),
            "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
            "category_id": 100,
            "id": int(re.findall("\d+", filename)[0]) * 10,
            "image_id": int(re.findall("\d+", filename)[0]),
            "iscrowd": 0,
            "segmentation": segmentation
        })

    dataset["images"] = images_list
    dataset["annotations"] = annotations_list

    with open(ann_out_file, "w") as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    create_roofai_dataset_coco_annotations()
