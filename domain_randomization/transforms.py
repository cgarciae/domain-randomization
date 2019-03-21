from .entity import Entity
from typing import List
import numpy as np
import cytoolz as cz
import cv2
import typing as tp
import numpy as np
from imgaug import augmenters as iaa
import domain_randomization as dr
from . import components
from . import utils
from functools import wraps


def require_keys(*keys):
    def _wrapper(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):

            if isinstance(args[0], Transform):
                row = args[1]
            else:
                row = args[0]

            for key in keys:
                _row = row
                for key in key.split("."):
                    assert key in _row
                    _row = _row[key]

            return f(*args, **kwargs)

        return _wrapped

    return _wrapper


class Transform:
    pass


class Compose(Transform):
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, row: tp.Dict) -> tp.Dict:

        for system in self.transforms:
            row = system(row)

        return row


class NoOp(Transform):
    def __call__(self, row: tp.Dict) -> tp.Dict:
        return row


class Resize:
    def __init__(self, objects=None, background=None):
        self.objects = objects
        self.background = background

        if self.objects:
            if isinstance(self.objects, int):
                self.objects = dict(
                    w=self.objects,
                    h=self.objects,
                )

            elif isinstance(self.objects, (list, tuple)):
                self.objects = dict(
                    w=self.objects[0],
                    h=self.objects[1],
                )

        if self.background:
            if isinstance(self.background, int):
                self.background = dict(
                    w=self.background,
                    h=self.background,
                )

            elif isinstance(self.background, (list, tuple)):
                self.background = dict(
                    w=self.background[0],
                    h=self.background[1],
                )

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        if self.objects:
            assert "objects" in row

            for obj in row["objects"]:
                obj: components.Object

                w = self.objects.get("w", obj.image.shape[1])
                h = self.objects.get("h", obj.image.shape[0])

                if hasattr(h, "__iter__"):
                    h = np.random.randint(low=h[0], high=h[1])

                if hasattr(w, "__iter__"):
                    w = np.random.randint(low=w[0], high=w[1])

                obj.image = cv2.resize(obj.image, (w, h))

        if self.background:

            background = row["background"]

            w = self.background.get("w", background.image.shape[1])
            h = self.background.get("h", background.image.shape[0])

            if hasattr(h, "__iter__"):
                h = np.random.randint(low=h[0], high=h[1])

            if hasattr(w, "__iter__"):
                w = np.random.randint(low=w[0], high=w[1])

            background.image = cv2.resize(background.image, (w, h))

        return row


class RandomBrightness:
    def __init__(self, objects_range=None, background_range=None):
        self.objects_range = objects_range
        self.background_range = background_range

        if self.objects_range and isinstance(self.objects_range, (int, float)):
            self.objects_range = (-self.objects_range, self.objects_range)

        if self.background_range and isinstance(self.background_range, (int, float)):
            self.background_range = (-self.background_range, self.background_range)

    def __call__(self, row: tp.Dict) -> tp.Dict:

        if self.objects_range:
            assert "objects" in row

            for obj in row["objects"]:
                obj: components.Object

                hsv = cv2.cvtColor(obj.image[..., :3], cv2.COLOR_RGB2HSV).astype(np.int32)  #convert it to hsv

                random_add = np.random.randint(self.objects_range[0], self.objects_range[1], dtype=np.int32)

                hsv[..., 2] = np.clip(
                    hsv[..., 2] + random_add,
                    0,
                    255,
                )

                hsv = hsv.astype(np.uint8)

                obj.image[..., :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if self.background_range:

            background = row["background"]

            hsv = cv2.cvtColor(background.image[..., :3], cv2.COLOR_RGB2HSV).astype(np.int32)  #convert it to hsv

            random_add = np.random.randint(self.background_range[0], self.background_range[1])

            hsv[..., 2] = np.clip(
                hsv[..., 2] + random_add,
                0,
                255,
            )

            hsv = hsv.astype(np.uint8)

            background.image[..., :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return row


class RandomChannelMultiply:
    def __init__(self, objects_range=None, background_range=None):
        self.objects_range = objects_range
        self.background_range = background_range

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        if self.objects_range:
            augmenter: iaa.Augmenter = iaa.Multiply(mul=self.objects_range, per_channel=True)

            for obj in row["objects"]:
                obj: components.Object

                obj.image[..., :3] = augmenter.augment_image(obj.image[..., :3])

        if self.background_range:
            augmenter: iaa.Augmenter = iaa.Multiply(mul=self.background_range, per_channel=True)

            background = row["background"]

            background.image[..., :3] = augmenter.augment_image(background.image[..., :3])

        return row


class RandomRotation90:
    def __init__(self, objects=False, background=False):
        self.objects = objects
        self.background = background
        self.angles = [0, 90, 180, 270]

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        if self.objects:
            for obj in row["objects"]:
                obj: components.Object
                angle = np.random.choice(self.angles)
                obj.image = utils.rotate_bound(obj.image, angle)

        if self.background:
            background = row["background"]
            angle = np.random.choice(self.angles)
            background.image = utils.rotate_bound(background.image, angle)

        return row


class RandomChannelInvert:
    def __init__(self, objects=False, background=False):
        self.objects = objects
        self.background = background

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        augmenter: iaa.Augmenter = iaa.Invert(p=0.5, per_channel=True)

        if self.objects:
            for obj in row["objects"]:
                obj: components.Object

                obj.image[..., :3] = augmenter.augment_image(obj.image[..., :3])

        if self.background:
            background = row["background"]

            background.image[..., :3] = augmenter.augment_image(background.image[..., :3])

        return row


class ObjectRandomScale:
    def __init__(self, scale=1.0):

        if not isinstance(scale, (list, tuple)):
            scale = (scale, 1.0 / scale)

        self.min = min(scale)
        self.max = max(scale)

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        for obj in row["objects"]:
            obj: components.Object

            scale = np.random.uniform(low=self.min, high=self.max)

            w = int(obj.image.shape[1] * scale)
            h = int(obj.image.shape[0] * scale)

            obj.image = cv2.resize(obj.image, (w, h))

        return row


class ObjectRandomAlphaChannelMultiply:
    def __init__(self, mult_range=(0.5, 2.0)):

        if not isinstance(mult_range, (list, tuple)):
            mult_range = (mult_range, 1.0 / mult_range)

        self.min = min(mult_range)
        self.max = max(mult_range)

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        for obj in row["objects"]:
            obj: components.Object

            scale = np.random.uniform(low=self.min, high=self.max)

            alpha = np.clip(obj.image[..., 3].astype(np.int32) * scale, 0, 255)
            obj.image[..., 3] = alpha.astype(np.uint8)

        return row


class ObjectRandomPosition:
    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        background: components.Background = row["background"]
        objects: tp.Iterable[components.Object] = row["objects"]

        image_h: int = background.image.shape[0]
        image_w: int = background.image.shape[1]

        xmin = self.xmin if self.xmin is not None else 0
        xmax = self.xmax if self.xmax is not None else image_w
        ymin = self.ymin if self.ymin is not None else 0
        ymax = self.ymax if self.ymax is not None else image_h

        for obj in objects:
            obj_h: int = obj.image.shape[0]
            obj_w: int = obj.image.shape[1]

            low = max(0, xmin)
            high = min(
                max(image_w - obj_w, 1),
                xmax,
            )
            high = max(high, low + 1)

            obj.x = np.random.randint(
                low=low,
                high=high,
            )

            low = max(0, ymin)
            high = min(
                max(image_h - obj_h, 1),
                ymax,
            )
            high = max(high, low + 1)

            obj.y = np.random.randint(
                low=low,
                high=high,
            )

        return row


class ObjectRandomRotation:
    def __init__(self, angles):
        if not hasattr(angles, "__iter__"):
            angles = (-angles, angles)
        self.angles = angles

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        for obj in row["objects"]:
            obj: components.Object

            angle = np.random.uniform(
                low=self.angles[0],
                high=self.angles[1],
            )

            obj.image = utils.rotate_bound(obj.image, angle)

        return row


class ObjectRandomPiecewiseAffine:
    def __init__(self, scale, nb_rows=4, nb_cols=4):
        self.aug = iaa.PiecewiseAffine(scale=scale, nb_rows=nb_rows, nb_cols=nb_cols)

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        for obj in row["objects"]:
            obj: components.Object
            obj.image = self.aug.augment_image(obj.image)

        return row


class NonMaxSupression:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        objects: tp.List[components.Object] = row["objects"]

        boxes = np.array([obj.bounding_box for obj in objects])
        scores = np.random.uniform(size=(len(boxes), ))
        valid_indexes = utils.non_max_suppression(
            boxes=boxes,
            scores=scores,
            iou_threshold=self.iou_threshold,
        )

        row["objects"] = [objects[i] for i in valid_indexes]

        return row


class GeneratePascalVoc:
    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        row["pascal_voq"] = components.PascalVoc(
            background=row["background"],
            objects=row["objects"],
        )

        return row


class GenerateSegmentation:
    def __init__(self, debug, hist_match=False, min_object_brightness=0):
        self.debug = debug
        self.hist_match = hist_match
        self.min_object_brightness = min_object_brightness

    def __call__(self, row: tp.Dict) -> tp.Dict:
        assert row, "row cannot be None"

        row["segmentation"] = components.Segmentation(
            background=row["background"],
            objects=row["objects"],
            hist_match=self.hist_match,
            min_object_brightness=self.min_object_brightness,
            debug=self.debug,
        )

        return row


class Sometimes(Transform):
    def __init__(self, prob, if_true, if_false=None):

        if if_false is None:
            if_false = NoOp()

        self.prob = prob
        self.if_true = if_true
        self.if_false = if_false

    def __call__(self, row):

        if self.prob < np.random.uniform():
            row = self.if_true(row)
        else:
            row = self.if_false(row)

        return row
