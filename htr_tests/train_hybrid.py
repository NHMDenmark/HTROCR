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
import utils.device as devutils
from PIL import Image
from itertools import groupby
import torch.utils.data as data
from functools import partial
import torch.nn.functional as F
from data_processors.emnist_datamodule import EMNISTDataModule

os.environ["CUDA_VISIBLE_DEVICES"]="0"
DATASET_PATH = "data"
CHECKPOINT_PATH = "./saved_models/"
device = devutils.default_device()

def train_hybrid(**kwargs):
    root_dir = os.path.join(CHECKPOINT_PATH, "nhmd_hybrid")
    data_module = EMNISTDataModule(train_bs=32, val_bs=32)
    data_module.setup()
    train_dl = data_module.train_dataloader()
    valid_dl = data_module.val_dataloader()
    # train_dl, valid_dl = prepare_data()
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
#                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=10,
                         gradient_clip_val=5)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need


    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "NHMD_hybrid.ckpt")
 #   if os.path.isfile(pretrained_filename):
 #       print("Found pretrained model, loading...")
 #       model = CNNTransformerHybrid.load_from_checkpoint(pretrained_filename)
 #   else:
    model = CNNTransformerHybrid(max_iters=trainer.max_epochs*len(train_dl), **kwargs)
    trainer.fit(model, train_dl, valid_dl)

    # Test best model on validation and test set
    val_result = trainer.test(model, valid_dl, verbose=False)
    print('val_result',val_result)
    result = {"val_loss": val_result[0]["test_loss"], "val_cer": val_result[0]["test_cer"], "val_acc": val_result[0]["test_acc"]}
#    result = {"val_loss": val_result[0]["test_loss"], "val_acc": val_result[0]["test_acc"]}
    
    model = model.to(device)
 #   trainer.save_checkpoint(os.path.join(CHECKPOINT_PATH, 'emnist.ckpt'))
    return model, result

model, result = train_hybrid(backbone_input_dim = 10,
                               backbone_fib_dim = 64,
                      backbone_expansion_factor = 8,
                                      model_dim = 512,
                                      num_heads = 8,
                                     num_layers = 3,
#                                     num_classes=10,
                                     vocab_size = 11,
                                             lr = 1e-4,
                                         warmup = 50,
                                        dropout = 0.0)

print(f"Validation CER:  {(100.0 * result['val_cer']):4.2f}%, Validation accuracy:  {(100.0 * result['val_acc']):4.2f}%")
#print(f"Validation accuracy:  {(100.0 * result['val_acc']):4.2f}%")

print(result)

train_dl, valid_dl = prepare_data()
number_of_test_imgs = 10
#test_loader = torch.utils.data.DataLoader(valid_dl, batch_size=number_of_test_imgs, shuffle=True)
test_preds = []
(x_test, y_test,_) = next(iter(valid_dl))
#inp_data = F.one_hot(x_test, num_classes=model.hparams.num_classes).float()
#inp_data = inp_data.to(device)
#preds = model(inp_data, add_positional_encoding=True)
#res = preds.argmax(dim=-1)
#print('actual', y_test[0])
#print('prediction', res[0])
y_pred = model(x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).cuda())
y_pred = y_pred.permute(1, 0, 2)
_, max_index = torch.max(y_pred, dim=2)
for i in range(x_test.shape[0]):
    raw_prediction = list(max_index[:, i].detach().cpu().numpy())
    prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != 10])
    test_preds.append("".join([str(s) for s in prediction.tolist()]))

for j in range(len(x_test)):
#    mpl.rcParams["font.size"] = 8
#    img = Image.fromarray(x_test[j]).save(f'test{j}.png')
    print("Actual: " + "".join([str(s) for s in y_test[j].cpu().numpy()]))
    print("Predicted: " + str(test_preds[j]))
