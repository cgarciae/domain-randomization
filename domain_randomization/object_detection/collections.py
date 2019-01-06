import numpy as np
import typing as tp
import domain_randomization as dr
from . import components

def create_collection(background: np.ndarray, objects: tp.List[tp.Tuple[str, np.ndarray]]):

        background = background
        background = dr.Entity([components.Background(background)])

        objects: tp.List[dr.Entity] = [
            dr.Entity([components.Object(image, label, x = 0, y = 0)])
            for label, image in objects
        ]

        return dr.Collection([background] + objects)