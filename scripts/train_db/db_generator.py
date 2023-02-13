import os
import numpy as np
from collectiontransformers.EmunchTransformer import EmunchTransformer
from collectiontransformers.MMDTransformer import MMDTransformer
from collectiontransformers.RetroTransformer import RetroTransformer
from collectiontransformers.TranscribusTransformer import TranscribusTransformer


def make_stratified_split(seed):
    """
    Selects random group of pregenerated samples out of all datasets based on given ratio
    """
    pass
    
def generate():
    pass


if __name__ == '__main__':
    # Prepare line level images
    # emunch_transformer = EmunchTransformer("/Users/linas/Studies/UCPH-DIKU/thesis/code/data/training_data/emunch")
    # emunch_transformer.process_collection()

    # mmd_transformer = MMDTransformer('/Users/linas/Studies/UCPH-DIKU/thesis/code/data/training_data/mmd')
    # mmd_transformer.process_collection()

    # retro_transformer = RetroTransformer('/Users/linas/Studies/UCPH-DIKU/thesis/code/data/training_data/retrodigitalisering/Fodby-Sogneråd-1842-1870')
    # retro_transformer.process_collection()

    transcribus_transformer = TranscribusTransformer('/Users/linas/Studies/UCPH-DIKU/thesis/code/data/training_data/Gjentofte_1881-1913_Denmark')
    transcribus_transformer.process_collection()