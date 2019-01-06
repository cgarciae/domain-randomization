from domain_randomization import object_detection as od
from domain_randomization import transforms
from domain_randomization import image_utils
import cv2
from matplotlib import pyplot as plt
import cytoolz as cz
from glob import glob
import random
import datetime

background = image_utils.inv_chanels(cv2.imread(
    "experiments/images/background.jpg", 
    cv2.IMREAD_UNCHANGED
))

dice_names = list(range(1, 7))
dice_labels = ["one", "two", "three", "four", "five", "six"]
dice_images = [ 
    image_utils.inv_chanels(cv2.imread(
        f"experiments/images/dice/{name}.png", 
        cv2.IMREAD_UNCHANGED,
    ))
    for name in dice_names
]

for i in range(10):
    dice_indexes = [
        random.choice(range(6))
        for i in range(10)
    ]
    dice = [
        (dice_labels[i], dice_images[i])
        for i in dice_indexes
    ]

    collection = od.create_collection(
        background = background,
        objects = dice,
    )

    transform = transforms.Compose([
        od.Resize(
            objects = dict(
                w = 150,
                h = 150,
            )
        ),
        od.RandomObjectPosition(),
        od.RandomObjectRotation(angles = 360),
        od.NonMaxSupression(0.0),
        od.GeneratePascalVoc(),
    ])

    collection = transform(collection)

    pv: od.PascalVoc = collection.first_component_of(od.PascalVoc)

    pv.save("experiments/images/results")
