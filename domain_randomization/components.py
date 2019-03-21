from domain_randomization import utils
from .pascal_voc_writer import PascalVocWritter
import uuid
import typing
import numpy as np
import os
import cv2
import PIL


class Object:
    def __init__(self, image: np.ndarray, label: typing.Union[str, int], x: int, y: int):
        self.image = image
        self.label = label
        self.x = x
        self.y = y

    @property
    def bounding_box(self):
        return [
            self.x,
            self.y,
            self.x + self.image.shape[1],
            self.y + self.image.shape[0],
        ]


class Background:
    def __init__(self, image: np.ndarray):
        self.image = image


class PascalVoc:
    def __init__(self, background: Background, objects: typing.List[Object]):

        image = background.image.copy()
        self.name = str(uuid.uuid4())

        pvw = PascalVocWritter(self.name + ".png", image.shape[1], image.shape[0])

        for obj in objects:
            image = utils.overlay_transparent(
                background=image,
                overlay=obj.image[..., :-1],
                mask=obj.image[..., -1],
                x=obj.x,
                y=obj.y,
            )
            xmin, ymin, xmax, ymax = obj.bounding_box

            pvw.add_object(obj.label, xmin, ymin, xmax, ymax)

        self.image = image
        self.metadata = pvw.metadata
        self.xml_string = pvw.xml_string

    def save(self, save_dir, extension="png"):
        image_path = os.path.join(save_dir, f"{self.name}.{extension}")
        xml_path = os.path.join(save_dir, f"{self.name}.xml")

        image = utils.inv_chanels(self.image)
        cv2.imwrite(image_path, image)

        with open(xml_path, mode="w") as f:
            f.write(self.xml_string)


class Segmentation:
    def __init__(self,
                 background: Background,
                 objects: typing.List[Object],
                 min_alpha=0,
                 hist_match=False,
                 min_object_brightness=0,
                 debug=False):

        image = background.image.copy()
        labels = np.zeros(image.shape[:2], dtype=np.uint8)

        for obj in objects:
            overlay = obj.image[..., :-1]
            mask = obj.image[..., -1]

            image = utils.overlay_transparent(
                background=image,
                overlay=overlay,
                mask=mask,
                x=obj.x,
                y=obj.y,
                hist_match=hist_match,
                min_object_brightness=min_object_brightness,
            )

            label_overlay = (np.ones(overlay.shape[:2]) * (obj.label + 1)).astype(np.uint8)
            label_mask = ((mask > min_alpha) * 255).astype(np.uint8)

            labels = utils.overlay_transparent(
                background=labels,
                overlay=label_overlay,
                mask=label_mask,
                x=obj.x,
                y=obj.y,
            )

        self.image = image
        self.labels = labels
        self.name = str(uuid.uuid4())
        self.debug = debug

    def save(self, save_dir, extension="png"):
        image_path = os.path.join(save_dir, f"{self.name}.{extension}")
        label_path = os.path.join(save_dir, f"{self.name}_label.png")
        debug_path = os.path.join(save_dir, f"{self.name}_debug.png")

        image = PIL.Image.fromarray(self.image)
        labels = PIL.Image.fromarray(self.labels)

        image.save(image_path)
        labels.save(label_path)

        if self.debug:
            segmentation_image = np.stack([labels] * 3, axis=-1)

            segmentation_image = ((segmentation_image > 0) * 255).astype(np.uint8)
            segmentation_image = (0.5 * self.image + 0.5 * segmentation_image).astype(np.uint8)

            segmentation_image = np.concatenate([self.image, segmentation_image], axis=1)

            segmentation = PIL.Image.fromarray(segmentation_image)
            segmentation.save(debug_path)


class Image:
    def __init__(self, image):
        self.image = image
