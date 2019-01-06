from domain_randomization import image_utils
from pascal_voc_writer import PascalVocWritter
import uuid
import typing
import numpy as np
import os
import cv2

class Object:
    
    def __init__(self, image: np.ndarray, label: str, x: int, y: int):
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
            image = image_utils.overlay_transparent(
                image,
                obj.image,
                obj.x,
                obj.y,
            )
            xmin, ymin, xmax, ymax = obj.bounding_box

            pvw.add_object(obj.label, xmin, ymin, xmax, ymax)

        self.image = image
        self.metadata = pvw.metadata
        self.xml_string = pvw.xml_string

    
    def save(self, save_dir):
        image_path = os.path.join(save_dir, self.name + ".png")
        xml_path = os.path.join(save_dir, self.name + ".xml")

        image = image_utils.inv_chanels(self.image)
        cv2.imwrite(image_path, image)

        with open(xml_path, mode = "w") as f:
            f.write(self.xml_string)