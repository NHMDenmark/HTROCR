import json
from line_segmentation.BaselineBuilder import BaselineBuilder
from abc import ABC, abstractmethod

class LineSegmenter(ABC):
    def __init__(self, path):
        with open(path) as f:
            self.config = json.load(f)
        self.bbuilder = BaselineBuilder(self.config)

    @abstractmethod
    def segment_lines(self, img):
        pass