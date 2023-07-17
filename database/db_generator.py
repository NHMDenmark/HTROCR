import os
from collectiontransformers.EmunchTransformer import EmunchTransformer
from collectiontransformers.MMDTransformer import MMDTransformer
from collectiontransformers.RetroTransformer import RetroTransformer
from collectiontransformers.TranscribusTransformer import TranscribusTransformer
import json
import argparse
import random
import shutil
    
def copy_images(lines, prefix, in_collection_path, out_image_path_dir):
    for line in lines:
        if in_collection_path == out_image_path_dir:
            continue
        file = line.split('\t')[0]
        if os.path.isdir(os.path.join(in_collection_path, 'image')):
            src_file = os.path.join(in_collection_path, 'image', file)
        else:
            src_file = os.path.join(in_collection_path, 'lines', file)
        dst_file = os.path.join(out_image_path_dir, prefix + file)
        shutil.copy(src_file, dst_file)

def split_validation_set(config):
    '''
    Makes train - validation set split from gt_train.txt,
    which contains all gathered samples.
    '''
    database_path = config['db_path']
    train_labels_file = 'gt_trainv2.txt'
    valid_labels_file = 'gt_valid.txt'
    with open(os.path.join(database_path, 'gt_train.txt'), 'r') as r:
        lines = r.readlines()
        random.shuffle(lines)
        val_set = random.sample(lines, 5000)
        train_set = [element for element in lines if element not in val_set]
    with open(os.path.join(database_path, train_labels_file), 'w') as w:
        w.write("".join(train_set))
    with open(os.path.join(database_path, valid_labels_file), 'w') as w:
        w.write("".join(val_set))

def generate(config):
    database_path = config['db_path']
    image_path_dir = os.path.join(database_path, 'image')
    os.makedirs(image_path_dir, exist_ok=True)
    train_labels_file = 'gt_train.txt'
    valid_labels_file = 'gt_valid.txt'
    db_collection = config['db_collection']
    collection_labels = []
    print("Number of collections", len(db_collection))
    im_files = os.listdir(image_path_dir)
    for index, subset in enumerate(db_collection):
        print("Running collection", index)
        subset_size = db_collection[subset]
        f = 'gt_train.txt'
        with open(os.path.join(subset, f), 'r') as r:
            lines = r.readlines()
            if subset_size != -1:
                lines = random.sample(lines, subset_size)
            collection_prefix = "d{}_".format(index)
            copy_images(lines, collection_prefix, subset, image_path_dir)
            collection_labels += [collection_prefix+l for l in lines if collection_prefix+l.split('\t')[0] in im_files]
    random.shuffle(collection_labels)
    val_set = random.sample(collection_labels, 10000)
    train_set = [element for element in collection_labels if element not in val_set]
    with open(os.path.join(database_path, train_labels_file), 'w') as w:
        w.write("".join(train_set))
    with open(os.path.join(database_path, valid_labels_file), 'w') as w:
        w.write("".join(val_set))

def prepare_line_level_images(config):
    # Prepare line level images from PAGE schema documents.
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
        # Assumes that all images in PAGE schema are placed into 'images' dir
        # at the same level as page xml dir
        transcribus_transformer = TranscribusTransformer(trankribus_path)
        transcribus_transformer.process_collection()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handwritten text line training set generator.')

    parser.add_argument('-p', '--config_path', help='Path to generator config', default='./config/generator.json')
    parser.add_argument('-l', '--generate_lines', help='Specify whether you want to generate lines', action=argparse.BooleanOptionalAction)
    parser.add_argument('-d', '--generate_db', help='Specify whether you want to generate database', action=argparse.BooleanOptionalAction)
    parser.add_argument('-s', '--split_db', help='Force another validation split', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    if args.generate_lines:
        prepare_line_level_images(config)
    if args.generate_db:
        generate(config)
    if args.split_db:
        split_validation_set(config)
