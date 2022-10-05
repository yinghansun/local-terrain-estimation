from enum import Enum

class PlaneLabel(Enum):
    horizontal = 'horizontal', 0, 'red'
    vertical = 'vertical', 1, 'blue'
    # sloping = 'sloping', 2, 'green'
    others = 'others', 2, 'yellow'

    def __init__(self, name: str, label: int, color: str) -> None:
        self.__name = name
        self.__label = label
        self.__color = color

    @property
    def name(self):
        return self.__name

    @property
    def label(self):
        return self.__label

    @property
    def color(self):
        return self.__color