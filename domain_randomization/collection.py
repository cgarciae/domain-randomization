

class Collection:

    def __init__(self, entities):
        self.entities = entities

    def entities_with(self, component_type):
        return filter(
            lambda e: e.has_component_type(component_type),
            self.entities,
        )

    