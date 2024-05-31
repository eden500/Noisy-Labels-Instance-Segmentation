import json
import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from shapely.geometry import Polygon
from tqdm import tqdm
import random
import argparse


def create_single_corruption_folder_copy_file(corruption_str, original_annotations_path, method_name):
    """
    Creates new folders and copies the original annotation file to the new folder for later modification.
    """
    base_path = f'noisy_data_{method_name}/coco_ann{corruption_str}'
    os.makedirs(f'{base_path}/train', exist_ok=True)
    os.makedirs(f'{base_path}/val', exist_ok=True)

    for split in ['train', 'val']:
        original_annotation_path = f'{original_annotations_path}/instances_{split}2017.json'
        destination_path = f'{base_path}/{split}'
        os.system(f'cp {original_annotation_path} {destination_path}')

    print(f'Created folder and copied annotation file for {corruption_str}')


def create_new_folders_and_copy_files(original_annotations_path, method_name, corruption_distances):
    """
    Creates new folders and copies files for the corrupted dataset.
    """
    name_to_add = f'_{method_name}_'

    for corruption_d in corruption_distances:
        if isinstance(corruption_d, dict):
            corruption_str = name_to_add + '_'.join(f'{k}_{str(v).replace(" ", "")}' for k, v in corruption_d.items()) + '_'
        else:
            corruption_str = name_to_add + str(corruption_d)

        create_single_corruption_folder_copy_file(corruption_str, original_annotations_path, method_name)

def pick_noise(probabilities):
    # Calculate the remaining probability for no noise.
    remaining_probability = 1 - sum(p[0] for p in probabilities.values())
    probabilities['none'] = [remaining_probability, None]

    keys = list(probabilities.keys())
    probs = [probabilities[key][0] for key in keys]
    chosen_key = random.choices(keys, weights=probs, k=1)[0]
    return chosen_key, probabilities[chosen_key][1]


def new_boundaries_with_prob(d, coco, ann_id, file, annotation):
    """
    Modifies the boundaries of an annotation with a given probability.
    """
    current_ann = coco.loadAnns(ann_id)[0]
    mask = coco.annToMask(current_ann)

    changed_class = False
    if 'flip_class' in d:
        boundary_ver = 'flip_class'
        cat_to_index_dict, index_to_cat_dict = get_index_cat_dicts(file)
        num_classes = get_num_classes(file)
        percent_noise = d.pop('flip_class')
        C = uniform_mix_C(percent_noise, num_classes) if boundary_ver == 'flip_class'\
            else uniform_assymetric_mix_C(percent_noise, file, cat_to_index_dict, num_classes) # to support assym later
        change_category_with_confusion_matrix_single(annotation, C, cat_to_index_dict, index_to_cat_dict)
        changed_class = annotation['category_id'] != current_ann['category_id']

    noise_type, k = pick_noise(d)

    if noise_type == 'rand':
        kernel = np.ones((k, k), np.uint8)
        new_mask = cv2.dilate(mask, kernel, iterations=1) if np.random.rand() < 0.5 else cv2.erode(mask, kernel, iterations=1)
    elif noise_type == 'localization':
        new_mask = add_localization_noise(mask, k)
    elif noise_type == 'approximation':
        new_mask = add_approximation_noise(mask, k)
    elif noise_type == 'none':
        new_mask = mask
    else:
        raise ValueError(f'Unknown boundary version: {noise_type}')

    # Convert modified mask back to RLE
    rle_modified = maskUtils.encode(np.asfortranarray(new_mask))
    rle_modified['counts'] = rle_modified['counts'].decode('utf-8') if isinstance(rle_modified['counts'], bytes) else rle_modified['counts']

    return rle_modified, noise_type, changed_class


def change_boundaries_for_file(file, coco, d, seed=1):
    """
    Changes the boundaries for all annotations in a given file.
    """
    if seed is not None:
        np.random.seed(seed)

    for annotation in tqdm(file['annotations']):
        new_mask, chosen_type, class_noise = new_boundaries_with_prob(d.copy(), coco, annotation['id'], file, annotation)
        if chosen_type != 'none':
            annotation['boundary_type'] = chosen_type
            annotation['segmentation'] = new_mask
        else:
            annotation['boundary_type'] = 'none'
        if class_noise:
            annotation['class_noise'] = True
        else:
            annotation['class_noise'] = False


def noise_annotations(method_name, corruption_distances, seed=1):
    """
    Adds noise to the annotations.
    """
    first_dir = f'noisy_data_{method_name}'
    name_to_add = f'_{method_name}_'

    for corruption_d in corruption_distances:
        corruption_str = name_to_add + ('_'.join(f'{k}_{str(v).replace(" ", "")}' for k, v in corruption_d.items()) + '_' if isinstance(corruption_d, dict) else str(corruption_d))

        for split in ['train', 'val']:
            with open(f'{first_dir}/coco_ann{corruption_str}/{split}/instances_{split}2017.json') as f:
                ann = json.load(f)

            coco = COCO(f'{first_dir}/coco_ann{corruption_str}/{split}/instances_{split}2017.json')
            change_boundaries_for_file(ann, coco, corruption_d, seed)

            with open(f'{first_dir}/coco_ann{corruption_str}/{split}/instances_{split}2017.json', 'w') as f:
                json.dump(ann, f)

        print(f'Finished {corruption_str}')


def mask_to_polygon(mask):
    """
    Converts a mask to a polygon.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour[:, 0, :] for contour in contours] if contours else []


def polygon_to_mask(polygon, h, w):
    """
    Converts a polygon to a mask.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def add_gaussian_noise(vertices, mean=0, std_dev=1):
    """
    Adds Gaussian noise to each vertex in a polygon.
    """
    noise = np.random.normal(mean, std_dev, vertices.shape)
    return np.round(vertices + noise).astype(int)


def add_localization_noise(mask, std_dev=1):
    """
    Adds Gaussian noise to the vertices of the polygon.
    """
    if np.sum(mask) == 0:
        return mask

    final_mask_noisy = np.zeros(mask.shape, dtype=np.uint8)
    for polygon in mask_to_polygon(mask):
        vertices_noisy = add_gaussian_noise(polygon, std_dev=std_dev)
        final_mask_noisy = np.maximum(final_mask_noisy, polygon_to_mask(vertices_noisy, mask.shape[0], mask.shape[1]))

    return final_mask_noisy


def simplify_polygon(polygon, tolerance):
    """
    Simplifies the polygon by removing vertices.
    """
    if len(polygon) < 4:
        return None

    shapely_polygon = Polygon(polygon)
    return shapely_polygon.simplify(tolerance, preserve_topology=True)


def simplified_polygon_to_mask(simplified_polygon, h, w):
    """
    Converts a simplified polygon back to a mask.
    """
    new_mask = np.zeros((h, w), dtype=np.uint8)
    simplified_coords = np.array(simplified_polygon.exterior.coords).reshape((-1, 1, 2))
    cv2.fillPoly(new_mask, [simplified_coords.astype(np.int32)], color=(1))
    return new_mask


def add_approximation_noise(mask, tolerance):
    """
    Adds noise to the vertices of the polygon by simplifying it.
    """
    if np.sum(mask) == 0:
        return mask

    final_mask_noisy = np.zeros(mask.shape, dtype=np.uint8)
    for polygon in mask_to_polygon(mask):
        simplified_polygon = simplify_polygon(polygon, tolerance)
        if simplified_polygon is None:
            continue
        mask_noisy = simplified_polygon_to_mask(simplified_polygon, mask.shape[0], mask.shape[1])
        final_mask_noisy = np.maximum(final_mask_noisy, mask_noisy)

    return final_mask_noisy


def change_category_with_confusion_matrix_single(annotation, C, cat_to_index_dict, index_to_cat_dict):
    """
    Changes the category_id of an annotation based on a confusion matrix.
    """
    chosen_index = np.random.choice(np.arange(C.shape[1]), p=C[cat_to_index_dict[annotation['category_id']]])
    annotation['category_id'] = index_to_cat_dict[chosen_index]


def get_index_cat_dicts(ann):
    """
    Returns dictionaries mapping category_id to index and vice versa.
    """
    cat_to_index_dict = {cat_info['id']: i for i, cat_info in enumerate(ann['categories'])}
    index_to_cat_dict = {i: cat_info['id'] for i, cat_info in enumerate(ann['categories'])}
    return cat_to_index_dict, index_to_cat_dict


def get_num_classes(ann):
    """
    Returns the number of classes in the dataset.
    """
    return len(ann['categories'])


def uniform_mix_C(mixing_ratio, num_classes):
    """
    Returns a linear interpolation of a uniform matrix and an identity matrix.
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + (1 - mixing_ratio) * np.eye(num_classes)


def uniform_assymetric_mix_C(mixing_ratio, ann, cat_to_index_dict, num_classes):
    """
    Returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row.
    """
    subcat_dict = {cat_info['supercategory']: [] for cat_info in ann['categories']}
    for cat_info in ann['categories']:
        subcat_dict[cat_info['supercategory']].append(cat_to_index_dict[cat_info['id']])

    C = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        C[i][subcat_dict[ann['categories'][i]['supercategory']]] = mixing_ratio * 1 / len(subcat_dict[ann['categories'][i]['supercategory']])
        C[i][i] += 1 - mixing_ratio
    return C


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotations_path', type=str, help='The path to the original annotations')
    parser.add_argument('--method_name', type=str, default='custom_method', help='The name of the method to be used for the corruption')
    parser.add_argument('--benchmark', type=str, default='', help='The benchmark to be used for the corruption')
    parser.add_argument('--corruption_values', type=list, default=[{'rand': [1, 3]}], help='The values to be used for the corruption')
    parser.add_argument('--seed', type=int, default=1, help='The seed to be used for the random noise')
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    Benchmarks = {
        'easy':
            {'approximation': [0.2, 5], 'localization': [0.2, 2], 'rand': [0.2, 3], 'flip_class': 0.2},
        'medium':
            {'approximation': [0.25, 10], 'localization': [0.25, 3], 'rand': [0.25, 5], 'flip_class': 0.3},
        'hard':
            {'approximation': [0.3, 15], 'localization': [0.3, 4], 'rand': [0.3, 7], 'flip_class': 0.4}
    }

    original_annotations_path = args.annotations_path
    corruption_values = args.corruption_values
    if args.benchmark:
        corruption_values = Benchmarks[args.benchmark]

    method_name = args.method_name
    if args.benchmark:
        method_name = args.benchmark

    create_new_folders_and_copy_files(original_annotations_path, method_name, corruption_values)
    noise_annotations(method_name, corruption_values, seed=args.seed)


# read me:

#  To run the benchmark, run the following:
#  python noise_annotations.py /path/to/annotations --benchmark {easy, medium, hard} (choose one of the three) --seed 1

#  To run a custom noise method, run the following:
#  python noise_annotations.py /path/to/annotations --method_name method_name
#  --corruption_values [{'rand': [scale_proportion, kernel_size(should be odd number)],
#  'localization': [scale_proportion, std_dev],
#  'approximation': [scale_proportion, tolerance], 'flip_class': percent_class_noise}]}]
#
#  for example:
#  python noise_annotations.py /path/to/annotations --method_name my_noise_method
#  --corruption_values [{'rand': [0.2, 3], 'localization': [0.2, 2], 'approximation': [0.2, 5], 'flip_class': 0.2}]
#
#  The script will create new folders and copy the original annotation files to the new folders. (the original annotations should be in coco format)
#  It will then add noise to the annotations in the new folders based on the specified corruption values.
#  The seed can be specified to ensure reproducibility.
