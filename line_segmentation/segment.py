import fire
from dependency_injector import containers, providers
from PreciseLineSegmenter import PreciseLineSegmenter
from HBLineSegmenter import HBLineSegmenter
from matplotlib import pyplot as plt

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    selector = providers.Selector(
        config.segmenter,
        precise=providers.Factory(PreciseLineSegmenter),
        height_based=providers.Factory(HBLineSegmenter),
    )

def run(path='./config/default.json'):
    container = Container()
    container.config.from_json(path)
    segmenter = container.selector(path)

    lines = segmenter.segment_lines("./demo/orig.jpg")

    plt.imshow(lines[17], cmap='gray')
    plt.show()

if __name__ == '__main__':
    fire.Fire(run)