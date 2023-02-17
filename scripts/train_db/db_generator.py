import os
import numpy as np
from collectiontransformers.EmunchTransformer import EmunchTransformer
from collectiontransformers.MMDTransformer import MMDTransformer
from collectiontransformers.RetroTransformer import RetroTransformer
from collectiontransformers.TranscribusTransformer import TranscribusTransformer
import sys


def make_stratified_split(seed):
    """
    Selects random group of pregenerated samples out of all datasets based on given ratio
    """
    pass
    
def generate():
    pass


if __name__ == '__main__':
    path = sys.argv[1]
    # Prepare line level images
    # emunch_transformer = EmunchTransformer(path)
    # emunch_transformer.process_collection()

    # mmd_transformer = MMDTransformer(path)
    # mmd_transformer.process_collection()

    # retro_transformer = RetroTransformer(path)
    # retro_transformer.process_collection()

    transcribus_transformer = TranscribusTransformer(path)
    transcribus_transformer.process_collection()