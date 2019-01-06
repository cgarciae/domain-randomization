from typing import List, Type, Dict

class Entity:

    def __init__(self, components: List):

        self.components: Dict[Type, object]  = {}

        for component in components:
            self.components[type(component)] = component
        
    # def add_component(self, component):
    #     component.entity = self
    #     self.components[type(component)] = component

    # def remove_component(self, component):
    #     component.entity = None
    #     del self.components[type(component)]

    def has_component_type(self, component_type):
        return any(map(lambda _class: _class == component_type, self.components.keys()))


