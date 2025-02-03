
import numpy as np
from openslide import OpenSlide, deepzoom


slide = 'data/NO 6407-6-5/mrxs/6407_6-5 1680 mDC.mrxs'

tile_generator = deepzoom.DeepZoomGenerator(
    OpenSlide(slide), tile_size=2048, overlap=0, limit_bounds=True
)

num_rows, num_cols = tile_generator.level_tiles[-1]

resolution = tile_generator.level_dimensions[-1]

downscale_factor = 12

images = []

for row in range(num_rows):
    for col in range(num_cols):
        tile = tile_generator.get_tile(tile_generator.level_count - 1, (col, row))
        images.append(tile)

new_res = (resolution[0] // downscale_factor, resolution[1] // downscale_factor)

thumbnail = OpenSlide(slide).get_thumbnail(new_res)

import matplotlib.pyplot as plt

# plot the thumbnail
plt.figure(figsize=(10, 10))

plt.imshow(thumbnail)
plt.axis('off')
plt.savefig('thumbnail.png', bbox_inches='tight', pad_inches=0, dpi=300)
# add horizontal lines
for i in range(1, num_rows):
    plt.axhline(i * 2048 // downscale_factor, color='r')
# add vertical lines
for i in range(1, num_cols):
    plt.axvline(i * 2048 // downscale_factor, color='r')
plt.savefig('thumbnail_with_grid.png', bbox_inches='tight', pad_inches=0, dpi=300)

tile = tile_generator.get_tile(tile_generator.level_count - 1, (19, 6))

from preprocessing.utils import get_boxes_yolo, compute_iou, get_boxes_thresh

boxes = get_boxes_yolo(tile)
iou = compute_iou(boxes) * (np.eye(len(boxes)) == 0)
non_overlapping_boxes = [box for i, box in enumerate(boxes) if np.all(iou[i] < 1e-3)]
overlapping_boxes = [box for i, box in enumerate(boxes) if np.any(iou[i] >= 1e-3)]

plt.figure(figsize=(10, 10))
plt.imshow(tile)
plt.axis('off')
plt.savefig('tile.png', bbox_inches='tight', pad_inches=0, dpi=300)
# add bounding boxes
for box in non_overlapping_boxes:
    dy, dx, y1, x1 = box
    plt.gca().add_patch(plt.Rectangle((dx, dy), x1 - dx, y1 - dy, linewidth=2, edgecolor='r', facecolor='none'))
plt.savefig('tile_with_boxes_thresh.png', bbox_inches='tight', pad_inches=0, dpi=300)
for box in overlapping_boxes:
    dy, dx, y1, x1 = box
    plt.gca().add_patch(plt.Rectangle((dx, dy), x1 - dx, y1 - dy, linewidth=2, edgecolor='r', facecolor='none'))
plt.savefig('tile_with_overlapping_boxes_thresh.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()

boxes = get_boxes_thresh(tile)