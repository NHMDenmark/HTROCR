import json
from BaselineBuilder import BaselineBuilder
from abc import ABC, abstractmethod

from skimage.color import rgb2gray

class LineSegmenter(ABC):
    def __init__(self, path="./config/default.json"):
        with open(path) as f:
            self.config = json.load(f)
        self.bbuilder = BaselineBuilder(self.config)

    @abstractmethod
    def segment_lines(self, img):
        pass