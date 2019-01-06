import fire
from glob import glob
import numpy as np
import cv2
import os

from . import transforms as tr
from . import object_detection as od
from . import image_utils

class API:

    def object_detection(self, 
        n_samples,
        n_objects,
        objects_pattern, 
        backgrounds_pattern,
        output_dir,
        rotation_angles = None,
        object_resize = None,
        background_resize = None,
        iou_threshold = 0.0,
    ):
        # create transform
        transform = tr.Compose([
            od.Resize(
                objects = object_resize,
                background = background_resize,
            ) if object_resize or background_resize else tr.NoOp(),
            od.RandomObjectPosition(),
            od.RandomObjectRotation(
                angles = rotation_angles,
            ) if rotation_angles else tr.NoOp(),
            od.NonMaxSupression(
                iou_threshold = iou_threshold,
            ),
            od.GeneratePascalVoc(),
        ])

        # get iterables
        all_object_filepaths = glob(objects_pattern)
        all_object_labels = [ 
            os.path.dirname(filepath).split(os.sep)[-1]
            for filepath in all_object_filepaths
         ]
        all_background_filepaths = glob(backgrounds_pattern)
        all_object_idx = np.arange(len(all_object_filepaths))

        # make output dir
        os.makedirs(output_dir, exist_ok=True)

        for _i in range(n_samples):
            if hasattr(n_objects, "__iter__"):
                n_objs = np.random.randint(low = n_objects[0], high = n_objects[1] + 1)
            else:
                n_objs = n_objects

            object_idxs = np.random.choice(all_object_idx, n_objs)

            object_images = [
                image_utils.inv_chanels(cv2.imread(
                    all_object_filepaths[i],
                    cv2.IMREAD_UNCHANGED,
                ))
                for i in object_idxs
            ]

            object_labels = [
                all_object_labels[i]
                for i in object_idxs
            ]

            background_idx = np.random.randint(len(all_background_filepaths))
            background_image = image_utils.inv_chanels(cv2.imread(
                all_background_filepaths[background_idx],
                cv2.IMREAD_UNCHANGED,
            ))

            collection = od.create_collection(
                background = background_image,
                objects = list(zip(object_labels, object_images))
            )

            collection = transform(collection)

            pascal_voc: od.PascalVoc = collection.first_component_of(od.PascalVoc)

            pascal_voc.save(output_dir)



def main():
    fire.Fire(API)