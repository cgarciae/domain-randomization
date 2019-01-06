
import image_utils

class Image:

    def __init__(self, image, position):
        self.image = image
        self.position = list(position)

    @property
    def bounding_box(self):
        return self.position + [ 
            self.position[0] + self.image.shape[1], 
            self.position[1] + self.image.shape[0],
        ]

class ObjectImage(Image):
    pass

class BackgroundImage(Image):

    def __init__(self, image):
        super().__init__(
            image = image, 
            position = [0, 0],
        )


class ObjectDetectionImage(Image):

    def __init__(self, background_image, object_images):

        image = background_image.image


        for object_image in object_images:
            image = image_utils.overlay_transparent(
                image,
                object_image.image,
                object_image.position[0],
                object_image.position[1],
            )

        

        super().__init__(
            image = image, 
            position = [0, 0],
        )