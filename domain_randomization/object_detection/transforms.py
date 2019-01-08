import numpy as np
import cytoolz as cz
import cv2
import typing as tp
import numpy as np
from imgaug import augmenters as iaa

import domain_randomization as dr
from . import components



class Resize:

    def __init__(self, objects = tp.Dict, background = tp.Dict):
        self.objects = objects
        self.background = background

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"

        if self.objects:
            if isinstance(self.objects, int):
                self.objects = dict(
                    w = self.objects,
                    h = self.objects, 
                )

            elif isinstance(self.objects, (list, tuple)):
                self.objects = dict(
                    w = self.objects[0],
                    h = self.objects[1],
                )
            

            for obj in collection.components_of(components.Object):
                obj: components.Object

                w = self.objects.get("w", obj.image.shape[1])
                h = self.objects.get("h", obj.image.shape[0])

                if hasattr(h, "__iter__"):
                    h = np.random.randint(low = h[0], high = h[1])
                
                if hasattr(w, "__iter__"):
                    w = np.random.randint(low = w[0], high = w[1])

                obj.image = cv2.resize(obj.image, (w, h))


        if self.background:
            if isinstance(self.background, int):
                self.background = dict(
                    w = self.background,
                    h = self.background, 
                )
            elif isinstance(self.background, (list, tuple)):
                self.background = dict(
                    w = self.background[0],
                    h = self.background[1],
                )

            background = collection.first_component_of(components.Background)

            w = self.background.get("w", background.image.shape[1])
            h = self.background.get("h", background.image.shape[0])

            if hasattr(h, "__iter__"):
                h = np.random.randint(low = h[0], high = h[1])
            
            if hasattr(w, "__iter__"):
                w = np.random.randint(low = w[0], high = w[1])

            background.image = cv2.resize(background.image, (w, h))

            
        return collection

class RandomChannelMultiply:

    def __init__(self, objects = False, background = False):
        self.objects = objects
        self.background = background

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"

        augmenter: iaa.Augmenter = iaa.Multiply(mul = (0.075, 1.0), per_channel=True)

        if self.objects:
            for obj in collection.components_of(components.Object):
                obj: components.Object

                obj.image[..., :3] = augmenter.augment_image(obj.image[..., :3])

        if self.background:
            background = collection.first_component_of(components.Background)

            background.image[..., :3] = augmenter.augment_image(background.image[..., :3])

            
        return collection

class RandomRotation90:

    def __init__(self, objects = False, background = False):
        self.objects = objects
        self.background = background
        self.angles = [0, 90, 180, 270]

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"

        if self.objects:
            for obj in collection.components_of(components.Object):
                obj: components.Object
                angle = np.random.choice(self.angles)
                obj.image = dr.image_utils.rotate_bound(obj.image, angle)

        if self.background:
            background = collection.first_component_of(components.Background)
            angle = np.random.choice(self.angles)
            background.image = dr.image_utils.rotate_bound(background.image, angle)
            

            
        return collection


class RandomChannelInvert:

    def __init__(self, objects = False, background = False):
        self.objects = objects
        self.background = background

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"

        augmenter: iaa.Augmenter = iaa.Invert(p = 0.5, per_channel = True)

        if self.objects:
            for obj in collection.components_of(components.Object):
                obj: components.Object

                obj.image[..., :3] = augmenter.augment_image(obj.image[..., :3])

        if self.background:
            background = collection.first_component_of(components.Background)

            background.image[..., :3] = augmenter.augment_image(background.image[..., :3])

            
        return collection

class ObjectRandomScale:

    def __init__(self, scale = 1.0):

        if not isinstance(scale, (list, tuple)):
            scale = (scale, 1.0 / scale)
            
        self.min = min(scale)
        self.max = max(scale)

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"


        for obj in collection.components_of(components.Object):
            obj: components.Object

            scale = np.random.uniform(low = self.min, high = self.max)

            w = int(obj.image.shape[1] * scale)
            h = int(obj.image.shape[0] * scale)

            obj.image = cv2.resize(obj.image, (w, h))
            
        return collection

class ObjectRandomPosition:

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"

        background: components.Background = cz.first(
            collection.components_of(components.Background)
        )
        objects: tp.Iterable[components.Object] = collection.components_of(components.Object)

        image_h: int = background.image.shape[0]
        image_w: int = background.image.shape[1]

        for obj in objects:
            obj_h: int = obj.image.shape[0]
            obj_w: int = obj.image.shape[1]

            obj.x = np.random.randint(
                low = 0,
                high = image_w - obj_w,
            )
            obj.y = np.random.randint(
                low = 0,
                high = image_h - obj_h,
            )

        return collection


class ObjectRandomRotation:

    def __init__(self, angles):
        if not hasattr(angles, "__iter__"):
            angles = (-angles, angles)
        self.angles = angles

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"

        for obj in collection.components_of(components.Object):
            obj: components.Object

            angle = np.random.uniform(
                low = self.angles[0],
                high = self.angles[1],
            )

            obj.image = dr.image_utils.rotate_bound(obj.image, angle)

        return collection

class NonMaxSupression:

    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"

        entities: tp.List[dr.Entity] = list(collection.entities_with(components.Object))
        objects: tp.List[components.Object] = [ entity.components[components.Object] for entity in entities ]

        boxes = np.array([
            obj.bounding_box
            for obj in objects
        ])
        scores = np.random.uniform(size = (len(boxes),))
        valid_indexes = dr.image_utils.non_max_suppression(
            boxes = boxes,
            scores = scores,
            iou_threshold = self.iou_threshold,
        )
        all_indexes = np.arange(len(boxes))
        filtered_indexes = all_indexes[~np.isin(all_indexes, valid_indexes)]

        for i in filtered_indexes:
            collection.remove(entities[i])

        return collection


    
class GeneratePascalVoc:

    def __call__(self, collection: dr.Collection = None) -> dr.Collection:
        assert collection, "collection cannot be None"


        pv = components.PascalVoc(
            background = collection.first_component_of(components.Background),
            objects = list(collection.components_of(components.Object)),
        )

        return collection.add(
            dr.Entity([pv]
        ))
