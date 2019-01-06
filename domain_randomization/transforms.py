
from domain_randomization import Collection
from typing import List

class Compose:

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, collection: Collection = None) -> Collection:

        for system in self.transforms:
            collection = system(collection)

        return collection


class NoOp:

    def __call__(self, collection: Collection = None) -> Collection:
        return collection