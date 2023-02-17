import os
from collectiontransformers.EmunchTransformer import EmunchTransformer
from collectiontransformers.MMDTransformer import MMDTransformer
from collectiontransformers.RetroTransformer import RetroTransformer
from collectiontransformers.TranscribusTransformer import TranscribusTransformer
import json
import argparse
import random
import shutil
    
def copy_images(lines, in_collection_path, out_image_path_dir):
    for line in lines:
        file = line.split('\t')[0]
        src_file = os.path.join(in_collection_path, file)
        shutil.copy(src_file, out_image_path_dir)

def generate(config):
    database_path = config['db_path']
    image_path_dir = os.path.join(database_path, 'image')
    os.makedirs(image_path_dir)
    labels_file = 'gt_train.txt'
    db_collection = config['db_collection']
    collection_labels = []
    for subset in db_collection:
        subset_size = db_collection[subset]
        lines = []
        with open(os.path.join(subset, 'gt_train.txt'), 'r') as r:
            lines = r.readlines()
        if subset_size != -1:
            lines = random.sample(lines, subset_size)
        copy_images(lines, subset, image_path_dir)
        collection_labels += lines
    with open(os.path.join(database_path, labels_file), 'w') as w:
        w.write("".join(collection_labels))

def prepare_line_level_images(config):
    # Prepare line level images
    emunch_path = config['transformers']['emunch']
    mmd_path = config['transformers']['mmd']
    retro_path = config['transformers']['retro']
    trankribus_path = config['transformers']['transkribus']
    if emunch_path:
        emunch_transformer = EmunchTransformer(emunch_path)
        emunch_transformer.process_collection()

    if mmd_path:
        mmd_transformer = MMDTransformer(mmd_path)
        mmd_transformer.process_collection()

    if retro_path:
        retro_transformer = RetroTransformer(retro_path)
        retro_transformer.process_collection()

    if trankribus_path:
        transcribus_transformer = TranscribusTransformer(trankribus_path)
        transcribus_transformer.process_collection()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handwritten text line training set generator.')

    parser.add_argument('-p', '--config_path', help='Path to generator config', default='./config/generator.json')
    parser.add_argument('-l', '--generate_lines', help='Specify whether you want to generate lines', action=argparse.BooleanOptionalAction)
    parser.add_argument('-d', '--generate_db', help='Specify whether you want to generate database', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    if args.generate_lines:
        prepare_line_level_images(config)
    if args.generate_db:
        generate(config)