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
            if total_mask_pixels == 0:
                # We only check whether the target label exists in the boxes
                label_boxes = [b for b, l in zip(boxes, labels) if l == label]
                if len(label_boxes) > 0:
                    # Sanity check: these boxes should be empty
                    for b in label_boxes:
                        Hmin, Wmin, Dmin, Hmax, Wmax, Dmax = map(int, b)
                        assert (Hmax - Hmin) * (Wmax - Wmin) * (Dmax - Dmin) == 0, \
                            f"Box with label {label} exists but mask pixels are zero"
                print(f"No mask pixels found for label {label}")
