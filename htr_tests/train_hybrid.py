import torch
import torch.nn as nn
from tqdm import tqdm
import utils.device as device
from data_processors.emnist_data_processor import prepare_data
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from hybrid_transformer import CNNTransformerHybrid

DATASET_PATH = "data"
CHECKPOINT_PATH = "./saved_models/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train_hybrid(**kwargs):
    root_dir = os.path.join(CHECKPOINT_PATH, "nhmd_hybrid")
    train_dl, valid_dl = prepare_data()
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=10,
                         gradient_clip_val=5)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "NHMD_hybrid.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = CNNTransformerHybrid.load_from_checkpoint(pretrained_filename)
    else:
        model = CNNTransformerHybrid(max_iters=trainer.max_epochs*len(train_dl), **kwargs)
        trainer.fit(model, train_dl, valid_dl)

    # Test best model on validation and test set
    val_result = trainer.test(model, valid_dl, verbose=False)
    result = {"val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result

reverse_model, reverse_result = train_hybrid(backbone_input_dim = 1,
                                                backbone_fib_dim = 64,
                                                backbone_expansion_factor = 8,
                                                model_dim = 256,
                                                num_heads = 4,
                                                num_layers = 16,
                                                vocab_size = 91,
                                                lr = 5e-4,
                                                warmup = 100,
                                                dropout=0.1)

print(f"Val accuracy:  {(100.0 * reverse_result['val_acc']):4.2f}%")