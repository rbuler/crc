import numpy as np

def check_masks_inside_boxes(data_loader, label):
    # This function checks if the mask pixels are inside the bounding boxes
    for images, targets in data_loader:
        for i in range(len(images)):
            print(targets[i]["id"], end=": ")
            labels, mask, boxes = targets[i]["labels"], targets[i]["mask"], targets[i]["boxes"]
            labels += 1
            total_mask_pixels = np.count_nonzero(mask==label)
            inside_pixels = 0
            bbox_volume = 0
            assert mask.shape == images[i].shape
            for box, label in zip(boxes, labels):
                Hmin, Wmin, Dmin, Hmax, Wmax, Dmax = map(int, box) # x is Height, y is Width, z is Depth
                mask_region = mask[:, Hmin:Hmax, Wmin:Wmax, Dmin:Dmax]
                bbox_volume += (Hmax - Hmin) * (Wmax - Wmin) * (Dmax - Dmin)
                inside_pixels += np.count_nonzero(mask_region == label)
                # print(f"Box: {box}, Label: {label}, Inside pixels: {inside_pixels}, Total mask pixels: {total_mask_pixels}")
            if total_mask_pixels > 0:
                percentage_inside = (inside_pixels / total_mask_pixels) * 100
                print(f"{percentage_inside:.2f}% of mask pixels are inside the bounding boxes")
                percentage_bbox_volume_occupied = (inside_pixels / bbox_volume) * 100 if bbox_volume > 0 else 0
                print(f"  {percentage_bbox_volume_occupied:.2f}% of the bounding box volume is occupied by the mask")
            else:
                print(f"No mask pixels found")
                assert inside_pixels == 0 and bbox_volume == 0 and len(labels) == 0 and int(boxes.sum().item()) == 0
