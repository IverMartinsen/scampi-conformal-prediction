import os
import argparse
import ast
import glob
import h5py
import time
import yolov5
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from openslide import OpenSlide, deepzoom
from utils import get_crop, compute_iou, get_boxes_yolo, get_boxes_thresh
import torch
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_slides", type=str, default='/Users/ima029/Desktop/NO 6407-6-5/mrxs')
parser.add_argument("--destination", type=str, default="./hdf5")
parser.add_argument("--method", type=str, default="yolov5")
parser.add_argument("--image_shape", type=tuple, default=(224, 224, 3))
args = parser.parse_args()


slides = glob.glob(args.path_to_slides + "/*.mrxs")
slides.sort()

print(f"Found {len(slides)} slides.")

model = yolov5.load("keremberke/yolov5m-aerial-sheep")  # pretrained model for sheep detection


if args.method == "yolov5":
    get_boxes = get_boxes_yolo
elif args.method == "thresh":
    get_boxes = get_boxes_thresh
else:
    raise ValueError("No such method")


def write_slide_to_hdf5(images, metadata, file_name):
    '''Write a slide to an hdf5 file'''
    file = h5py.File(file_name, "w")
    images = np.array(images, dtype=np.uint8)
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    metadata = np.array([str(m) for m in metadata], dtype="S")
    meta_dataset = file.create_dataset(
        "metadata", np.shape(metadata), data=metadata
    )
    file.close()


def read_slide_from_hdf5(file_name):
    '''Read a slide from an hdf5 file'''
    file = h5py.File(file_name, "r")

    images = np.array(file["/images"]).astype("uint8")
    metadata = np.array(file["/metadata"]).astype("str")
    
    metadata = [ast.literal_eval(m) for m in metadata]
    
    return images, metadata


if __name__ == "__main__":

    os.makedirs(args.destination, exist_ok=True)

    for slide in tqdm(slides, total=len(slides)):
        
        file_name = os.path.join(args.destination, f"{os.path.basename(slide).split('.')[0]}.hdf5")
        if os.path.exists(file_name):
            print(f"Skipping {slide}.")
            continue
        
        start = time.time()
        
        tile_generator = deepzoom.DeepZoomGenerator(
            OpenSlide(slide), tile_size=2048, overlap=0, limit_bounds=True
        )
        
        num_rows, num_cols = tile_generator.level_tiles[-1]
        
        crops = []
        metadata = []
        
        for row in tqdm(range(num_rows), total=num_rows):
            for col in range(num_cols):
                tile = tile_generator.get_tile(tile_generator.level_count - 1, (row, col))
                boxes = get_boxes(tile)
                # only keep non-overlapping boxes
                if len(boxes) > 1: # if there are more than one box
                    try:
                        iou = compute_iou(boxes) * (np.eye(len(boxes)) == 0)
                    except TypeError:
                        print(boxes)
                        raise
                    # only keep non-overlapping boxes
                    boxes = [box for i, box in enumerate(boxes) if np.all(iou[i] < 1e-3)]
                for box in boxes:
                    crop = get_crop(box, tile, pad_image=False) # numpy array of variable resolution
                    crop = torchvision.io.encode_jpeg(torch.tensor(crop).permute(2, 0, 1))
                    # convert to uint8
                    #crop = tf.convert_to_tensor(crop, dtype=tf.uint8)
                    #print(crop)
                    #crop = tf.image.resize(crop, args.image_shape[:2])
                    #crop = tf.cast(crop, tf.uint8)
                    
                    address = {'slide': slide, 'row': row, 'col': col, 'box': list(box)}
                    
                    crops.append(crop)
                    metadata.append(address)
        
        file_name = os.path.join(args.destination, f"{os.path.basename(slide).split('.')[0]}.hdf5")
        #write_slide_to_hdf5(crops, metadata, file_name)
        with h5py.File(file_name, "w") as file:
            for i, (crop, meta) in enumerate(zip(crops, metadata)):
                file.create_dataset(f"{os.path.basename(slide).split('.')[0]}_crop_{i}", data=crop, dtype='uint8')
                #file.create_dataset(f"{os.path.basename(slide).split('.')[0]}_meta_{i}", data=str(meta))
        
        end = time.time()
        
        print(f"Extracted {len(crops)} crops from {slide} in {end - start} seconds.")
        