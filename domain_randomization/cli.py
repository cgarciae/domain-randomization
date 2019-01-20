import fire
from glob import glob
import numpy as np
import cv2
import os
from pypeln import process as pr
from tqdm import tqdm
import random

import domain_randomization as dr

class API:

    def object_detection(self, 
        n_samples,
        n_objects,
        objects_pattern, 
        backgrounds_pattern,
        output_dir,
        workers = 1,
        rotation_angles = None,
        object_resize = None,
        background_resize = None,
        iou_threshold = 0.0,
        object_channel_multiply = False,
        background_channel_multiply = False,
        object_channel_invert = False,
        background_channel_invert = False,
        background_rotate = False,
        object_scale = 1.0,
        output_extension = "png",
    ):
        
        # create transform
        transform = dr.Compose([
            dr.RandomChannelMultiply(
                objects = object_channel_multiply,
                background = background_channel_multiply,
            ) if object_channel_multiply or background_channel_multiply else dr.NoOp(),
            dr.RandomChannelInvert(
                objects = object_channel_invert,
                background = background_channel_invert,
            ) if object_channel_invert or background_channel_invert else dr.NoOp(),
            dr.Resize(
                objects = object_resize,
                background = background_resize,
            ) if object_resize or background_resize else dr.NoOp(),
            dr.RandomRotation90(
                background = background_rotate,
            ),
            dr.ObjectRandomPosition(),
            dr.ObjectRandomScale(
                scale = object_scale,
            ) if object_scale != 1.0 else dr.NoOp(),
            dr.ObjectRandomRotation(
                angles = rotation_angles,
            ) if rotation_angles else dr.NoOp(),
            dr.NonMaxSupression(
                iou_threshold = iou_threshold,
            ),
            dr.GeneratePascalVoc(),
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

        def create_sample(_i):
            if hasattr(n_objects, "__iter__"):
                n_objs = np.random.randint(low = n_objects[0], high = n_objects[1] + 1)
            else:
                n_objs = n_objects

            object_idxs = np.random.choice(all_object_idx, n_objs)

            object_images = [
                dr.image_utils.inv_chanels(cv2.imread(
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
            background_image = dr.image_utils.inv_chanels(cv2.imread(
                all_background_filepaths[background_idx],
                cv2.IMREAD_UNCHANGED,
            ))

            collection = dr.create_collection(
                background = background_image,
                objects = list(zip(object_labels, object_images))
            )

            collection = transform(collection)

            pascal_voc: dr.PascalVoc = collection.first_component_of(dr.PascalVoc)

            pascal_voc.save(output_dir, extension = output_extension)


        def on_start(worker_info):
            np.random.seed(worker_info.index)
            random.seed(worker_info.index + 100)
        
        stage = pr.map(create_sample, range(n_samples), workers=workers, on_start=on_start)
        stage = ( x for x in tqdm(stage, total=n_samples))

        pr.run(stage)



def main():
    fire.Fire(API)