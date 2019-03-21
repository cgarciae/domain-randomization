import cv2
import numpy as np


def inv_chanels(image):
    image[..., :3] = image[..., (2, 1, 0)]
    return image


def rotate_bound(image, angle):
    # add alpha channel if not present
    if image.shape[2] < 4:
        image = np.concatenate(
            [image, np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255],
            axis=2,
        )

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # distance matrix
    # dst_mat = np.zeros((h, w, 4), np.uint8)

    # perform the actual rotation and return the image
    return cv2.warpAffine(
        image,
        M,
        (nW, nH),
        # dst_mat,
        # flags=cv2.INTER_LINEAR,
        # borderMode=cv2.BORDER_TRANSPARENT,
    )


def overlay_transparent(background: np.ndarray,
                        overlay: np.ndarray,
                        mask: np.ndarray,
                        x: int,
                        y: int,
                        hist_match=False,
                        min_object_brightness=0):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]
        mask = mask[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]
        mask = mask[:h]

    mask = mask / 255.0

    if mask.ndim < background.ndim:
        mask = np.expand_dims(mask, axis=-1)

    if overlay.ndim < background.ndim:
        overlay = np.expand_dims(overlay, axis=-1)

    if background.ndim > 2 and background.shape[2] > 3:
        background = background[..., :3]

    if hist_match:
        overlay = histogram_matching_h(overlay, background[y:y + h, x:x + w], min_v=min_object_brightness)
    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay

    return background


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    assert boxes.shape[0] == scores.shape[0]

    if len(scores) == 0:
        return np.array([])

    # bottom-left origin
    ys1 = boxes[:, 1]
    xs1 = boxes[:, 0]
    # top-right target
    ys2 = boxes[:, 3]
    xs2 = boxes[:, 2]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    all_filtered = set()
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
        filtered_indexes = set((ious > iou_threshold).nonzero()[0])
        all_filtered |= filtered_indexes
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [v for (i, v) in enumerate(scores_indexes) if i not in filtered_indexes]

    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections.astype(np.float32) / unions

    return ious


def histogram_matching_h(src, target, min_v=0):
    # http://vzaguskin.github.io/histmatching1/
    src_hsv = cv2.cvtColor(src[..., :3], cv2.COLOR_RGB2HSV)
    target_hsv = cv2.cvtColor(target[..., :3], cv2.COLOR_RGB2HSV)
    num_bins = 255

    src_hist, bins = np.histogram(src_hsv[:, :, 2].flatten(), num_bins, density=True)
    target_hist, bins = np.histogram(target_hsv[:, :, 2].flatten(), num_bins, density=True)
    src_cdf = src_hist.cumsum()
    src_cdf = (255 * src_cdf / src_cdf[-1]).astype(np.uint8)
    target_cdf = target_hist.cumsum()
    target_cdf = (255 * target_cdf / target_cdf[-1]).astype(np.uint8)

    interp1 = np.interp(src_hsv[:, :, 2].flatten(), bins[:-1], src_cdf)
    interp2 = np.interp(interp1, target_cdf, bins[:-1])
    src_hsv[:, :, 2] = interp2.reshape(src_hsv.shape[0], src_hsv.shape[1])
    src_hsv[:, :, 2] = np.maximum(min_v, src_hsv[:, :, 2])

    res = src.copy()
    res[..., :3] = cv2.cvtColor(src_hsv, cv2.COLOR_HSV2RGB)
    return res
