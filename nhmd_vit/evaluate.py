import torch
import os
from tqdm import tqdm
from nhmd_vit.VitCTC import ViTCTC
import nhmd_vit.utils.device as devutils
from itertools import groupby
from nhmd_vit.data_processors.nhmd_datamodule import NHMDDataModule
from nhmd_vit.nhmdtokenizer import NHMDTokenizer
from transformers import RobertaTokenizer
from torchmetrics import CharErrorRate, WordErrorRate
import numpy as np

DATASET_PATH = "../unilm/trocr/NHMD_GT"
CHECKPOINT_PATH = "./nhmd_vit/saved_models/"
device = devutils.default_device()

# run_name = 'TROCR_beit_base_271k'
# cp_file = 'TROCR_beit_base_271k_final.ckpt'

# run_name = 'NHMD_deit_small_271k'
# cp_file = 'deit_small.ckpt'

# run_name = 'TROCR_beit_large_271k'
# cp_file = 'TROCR_beit_large_271k_final.ckpt'

run_name = 'TROCR_deit_small_271k_trocrinit'
cp_file = 'TROCR_deit_small_271k_trocrinit_final.ckpt'

def test_hybrid():
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer = NHMDTokenizer()
    data_module = NHMDDataModule(
        data_path=DATASET_PATH,
        tokenizer=tokenizer, 
        max_len=300,
        test_bs=1,
        num_workers=16)
    data_module.setup(mode='test')
    test_dl = data_module.test_dataloader()

    pretrained_filename = os.path.join(CHECKPOINT_PATH, run_name, cp_file)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ViTCTC.load_from_checkpoint(pretrained_filename)
        model.to(device)
    else:
        print('No checkpoint detected!')
        return
    cer = CharErrorRate()
    wer = WordErrorRate()
    results = []
    cer_scores = []
    wer_scores = []
    for batch in tqdm(test_dl):
        (x_test, y_test, attn_images) = batch
        pred = model(x_test.to(device), 50)
        pred = pred.permute(1, 0, 2)
        logits = pred.log_softmax(2)
        _, max_index = torch.max(logits, dim=2)
    
        # Loop through the batch images
        for i in range(x_test.shape[0]):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction_idxs = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != 0])
            try:
                # try to stop at eos_token_id for a specific line in batch
                idx = prediction_idxs.index(1)
            except:
                decoded_text = prediction_idxs
            else:
                decoded_text = prediction_idxs[: idx + 1]
            # Label text
            pred_str = tokenizer.decode(decoded_text, skip_special_tokens=True)
            label_str = tokenizer.decode(y_test[i], skip_special_tokens=True)
            results.append(str(label_str)+'\n')
            results.append(str(pred_str)+'\n')
            results.append('\n')
            cer_scores.append(cer(pred_str.lower(), label_str.lower()))
            wer_scores.append(wer(pred_str.lower(), label_str.lower()))

    mean_cer = np.array(cer_scores).mean()
    mean_wer = np.array(wer_scores).mean()
    results.append('mean cer: ' + str(mean_cer) + '\n')
    results.append('mean wer: ' + str(mean_wer))
    with open(os.path.join(CHECKPOINT_PATH,f'results_{run_name}.txt'), 'w') as f:
        f.write(''.join(results))

if __name__ == '__main__':
    test_hybrid()

    
