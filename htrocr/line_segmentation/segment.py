import fire
import os
from dependency_injector import containers, providers
from htrocr.line_segmentation.PreciseLineSegmenter import PreciseLineSegmenter
from htrocr.line_segmentation.HBLineSegmenter import HBLineSegmenter
from htrocr.line_segmentation.BaselineBuilder import BaselineBuilder
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
import numpy as np
import json

"""
Test script to play around with segmenations.
Use 'fire' for argument parsing. See their documentation for details.
"""
class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    selector = providers.Selector(
        config.segmenter,
        precise=providers.Factory(PreciseLineSegmenter),
        height_based=providers.Factory(HBLineSegmenter),
    )

def get_baselines(dir_path):
    with open('./config/default.json') as f:
        config = json.load(f)
    bbuilder = BaselineBuilder(config)
    for file in os.listdir(dir_path):
        orig = Image.open(os.path.join(dir_path, file))
        img = rgb2gray(np.array(orig))
        clusters, N = bbuilder.run(img)

def run(path='./config/default.json'):
    container = Container()
    container.config.from_json(path)
    segmenter = container.selector(path)

    lines = segmenter.segment_lines("./demo/orig.jpg")

    img = Image.fromarray(lines[17]*255).convert('L')
    img.save('test.jpg')
    # plt.imshow(lines[17], cmap='gray')
    # plt.show()

if __name__ == '__main__':
    fire.Fire(run)