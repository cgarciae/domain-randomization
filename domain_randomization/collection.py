from .entity import Entity
from typing import List, Iterable
import cytoolz as cz
import numpy as np
import typing as tp
from . import components


def create_scene(background: np.ndarray, objects: tp.List[tp.Tuple[str, np.ndarray]]):

    row = dict()

    row["background"] = components.Background(background)
    row["objects"]: tp.List[Entity] = [components.Object(
        image,
        label,
        x=0,
        y=0,
    ) for label, image in objects]

    return row
