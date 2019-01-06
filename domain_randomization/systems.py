


class Compose:

    def __init__(self, systems):
        self.systems = systems

    def __call__(self, collection):

        for system in self.systems:
            collection = system(collection)

        return collection


