from NHMDDataset import NHMDDataset
from TrOCREDProcessor import get_processor
from torch.utils.data import DataLoader, Subset
from transformers import VisionEncoderDecoderModel,TFVisionEncoderDecoderModel, AutoModel
import torch
import numpy as np
from PIL import Image 
from tqdm import tqdm
from datasets import load_metric

def run_evaluate():
    max_length = 300
    encoder_name = 'facebook/deit-tiny-patch16-224'
    decoder_name = 'xlm-roberta-base'

    processor = get_processor()
    ds = NHMDDataset("../../unilm/trocr/NHMD_GT", "test", processor, max_length, augment=False)
#    ds = NHMDDataset("../data/NHMD_train", "valid", processor, 300, False)
    subset = Subset(ds,range(16))
    dataloader = DataLoader(ds, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("./out/nhmd_small_ft_e", "./out/nhmd_small_ft_d")
    model.to(device)

    cer = load_metric('cer')

    preds = ""
    for batch in tqdm(dataloader):
         pixel_values = batch["pixel_values"].to(device)
         outputs = model.generate(pixel_values)

         # decode
         pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
         labels = batch["labels"]
         labels[labels == -100] = processor.tokenizer.pad_token_id
         label_str = processor.batch_decode(labels, skip_special_tokens=True)

         pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
         label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]
         
         # add batch to metric
         cer.add_batch(predictions=[s.lower() for s in pred_str], references=[s.lower() for s in label_str])

         preds += f"{pred_str}\t{label_str}\n"

    final_score = cer.compute()
    print(final_score)
    with open("results.txt", "w") as f:
        f.write(preds)

if __name__ == '__main__':
    run_evaluate()
