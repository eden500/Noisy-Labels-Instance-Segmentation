# TODO: remember to set multiple mask output to False when segmenting using sam.

import cv2
import pycocotools.coco as COCO
import pycocotools.mask as maskUtils
import torch
import numpy as np
import os
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import json
import argparse


def extract_bounding_box_and_area(segmentation, omega=1):
    """
    Extracts the bounding box and area for a single object from a boolean segmentation output.

    :param segmentation: A 2D boolean array where True represents the object.
    :return: A tuple containing the bounding box (x_min, y_min, x_max, y_max) and the area.
    """
    # Find indices where segmentation is True
    rows, cols = np.where(segmentation)

    # Calculate the area as the count of True values
    area = np.sum(segmentation)

    # If there are no True values, return an empty bounding box and zero area
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, 0, 0), 0

    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)
    
    # scale the size of the bounding box by multiplying the length and width by omega
    if omega != 1:
        x_min, x_max = max(x_min - int((omega-1)/2 * (x_max - x_min)), 0), min(x_max + int((omega-1)/2 * (x_max - x_min)), segmentation.shape[1])
        y_min, y_max = max(y_min - int((omega-1)/2 * (y_max - y_min)), 0), min(y_max + int((omega-1)/2 * (y_max - y_min)), segmentation.shape[0])
    

    bounding_box = (x_min, y_min, x_max, y_max)

    return bounding_box, area


def get_center_point(box, add_gaussian_noise=False, std_mult=3):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    if add_gaussian_noise:
        # set the std to be half the highy of box for x and half the width of box for y
        std_x = (box[2] - box[0]) / std_mult
        std_y = (box[3] - box[1]) / std_mult
        x_center += np.random.normal(0, std_x)
        y_center += np.random.normal(0, std_y)
        # make sure its within the box
        x_center = min(max(x_center, box[0]), box[2])
        y_center = min(max(y_center, box[1]), box[3])
    return np.array([[x_center, y_center]])


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        center_point = get_center_point(box, add_gaussian_noise=False)
        point_labels = np.array([1])  # Assuming the center point is a foreground point
        masks, scores, logits = sam_predictor.predict(
            point_coords=center_point,
            point_labels=point_labels,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_new_ann(annotation, mask, counter):
    '''
    annotation: a coco annotation
    sam_mask: a sam output mask
    '''
    new_ann = {}
    new_ann['image_id'] = annotation['image_id']
    new_ann['category_id'] = annotation['category_id']
    new_ann['segmentation'] = maskUtils.encode(np.asfortranarray(mask))
    if isinstance(new_ann['segmentation']['counts'], bytes):
        # Convert bytes to string
        new_ann['segmentation']['counts'] = new_ann['segmentation']['counts'].decode('utf-8')
    new_ann['bbox'], new_ann['area'] = extract_bounding_box_and_area(mask)
    # change bbox to xywh
    new_ann['bbox'] = [new_ann['bbox'][0], new_ann['bbox'][1], new_ann['bbox'][2] - new_ann['bbox'][0], new_ann['bbox'][3] - new_ann['bbox'][1]]
    new_ann['iscrowd'] = annotation['iscrowd']
    new_ann['id'] = counter
    return new_ann


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data', help='The path to the folder that contains the coco folders')
    parser.add_argument('--annotations_path', type=str, default='data/coco_ann2017/annotations', help='The path to the original annotations')
    parser.add_argument('--sam_path', type=str, default=os.path.join("weights", "sam_vit_h_4b8939.pth"), help='The path to the sam checkpoint')
    parser.add_argument('--scale_bbox', type=int, default=1, help='The values to be used for the scaling the bbox')
    return parser.parse_args()


# get args
args = argument_parser()

# set up cuda str
cuda_str = 'cuda:0'

DEVICE = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
print("DEVICE:", DEVICE)

# set up scale hyperparameter
omega_scale = args.scale_bbox

# set up sam checkpoint path
SAM_CHECKPOINT_PATH = args.sam_path
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

# load the sam model
SAM_ENCODER_VERSION = "vit_h"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)


for type in ['train', 'val']:
    # set up coco
    file_orig = f'{args.annotations_path}/instances_{type}2017.json'
    coco_orig = COCO.COCO(file_orig)

    # create the directory for the noisy data
    method_name = 'gt_plus_sam_point'
    method_name = method_name + '_omega' + str(omega_scale)
    corruption_str = '_auto_ann'
    os.makedirs('noisy_data_' + method_name + '/coco_ann' + corruption_str + '/' + type, exist_ok=True)

    # copy the original annotations to the noisy data directory
    os.system(f'cp {args.annotations_path}/instances_' + type + '2017.json noisy_data_' + method_name + '/coco_ann' + corruption_str + '/' + type)

    # load the annotations file we copied as json
    file = 'noisy_data_' + method_name + '/coco_ann' + corruption_str + '/'+ type + '/instances_' + type + '2017.json'
    loaded_json = json.load(open(file))

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.int64, np.uint64)):
                return int(obj)
            return json.JSONEncoder.default(self, obj)

    # set up images path
    images_path = f'{args.data_path}/coco_' + type + '2017/' + type + '2017/'

    # get the images from coco
    img_ids = coco_orig.getImgIds()

    # initialize the new annotations list
    counter = 0
    new_anns = []
    for img in tqdm(img_ids):
        # get the image info
        img_info = coco_orig.loadImgs(img)[0]
        img_name, img_id = img_info['file_name'], img_info['id']
        img_path = images_path + img_name
        # load image
        image = cv2.imread(img_path)
        
        # Get all annotations for the specified image
        annotations = coco_orig.loadAnns(coco_orig.getAnnIds(imgIds=img_id))
        
        # get the coordinates for bounding boxes in the image
        xyxy_boxes = []
        for annotation in annotations:
            segmentation = coco_orig.annToMask(annotation)
            xyxy, area = extract_bounding_box_and_area(segmentation, omega_scale)
            xyxy_boxes.append(np.array(xyxy))
        xyxy_boxes = np.array(xyxy_boxes)
        
        # convert detections to masks
        masks = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=xyxy_boxes
        )
        
        for seg, mask in zip(annotations, masks):
            new_ann = get_new_ann(seg, mask, counter)
            counter += 1
            new_anns.append(new_ann)
            
    # save the new annotations
    loaded_json['annotations'] = new_anns
    with open(file, 'w') as f:
        json.dump(loaded_json, f, cls=NumpyEncoder)
        print(f"Saved {file}")
    
    
    
    
    





