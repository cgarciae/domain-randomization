


class Entity:

    def __init__(self, components):

        self.components = {}

        for c in components:
            self.add_component(c)
        
    def add_component(self, component):
        component.entity = self
        self.components[type(component)] = component

    def remove_component(self, component):
        component.entity = None
        del self.components[type(component)]

    def has_component_type(self, component_type):
        return any(map(lambda _class: _class == component_type, self.components.keys()))


