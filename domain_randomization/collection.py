
from .entity import Entity
from typing import List, Iterable
import cytoolz as cz

class Collection:

    def __init__(self, entities: List[Entity]):
        self.entities = entities

    def add(self, entity: Entity):
        self.entities.append(entity)
        return self

    def remove(self, entity: Entity):
        self.entities.remove(entity)
        return self

    def entities_with(self, component_type) -> Iterable[Entity]:
        return filter(
            lambda e: e.has_component_type(component_type),
            self.entities,
        )

    def first_entity_with(self, component_type) -> Entity:
        return cz.first(self.entities_with(component_type))

    def components_of(self, component_type) -> Iterable[object]:
        components = filter(
            lambda e: e.has_component_type(component_type),
            self.entities,
        )

        components = map(
            lambda entity: entity.components[component_type],
            components,
        )

        return components

    def first_component_of(self, component_type) -> object:
        return cz.first(self.components_of(component_type))

    