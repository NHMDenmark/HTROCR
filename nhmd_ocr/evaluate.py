from nhmd_ocr.NHMDDataset import NHMDDataset
from nhmd_ocr.TrOCREDProcessor import get_processor
from torch.utils.data import DataLoader, Subset
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, AutoModel
import torch
from tqdm import tqdm
from datasets import load_metric
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
def run_evaluate():
    max_length = 300
    encoder_name = 'facebook/deit-tiny-patch16-224'
    decoder_name = 'xlm-roberta-base'

    processor = get_processor("microsoft/trocr-base-handwritten", 'xlm-roberta-base')
#    processor = get_processor("microsoft/trocr-base-handwritten", './nhmd_ocr/nhmd_out/nhmd_base_roberta_271k_d')
#    processor = get_processor("microsoft/trocr-base-handwritten", "./nhmd_ocr/nhmd_out/base_roberta/nhmd_base_roberta_final")
#    processor = get_processor("microsoft/trocr-base-handwritten")
#    processor = get_processor("microsoft/trocr-base-handwritten")
#    processor = get_processor("microsoft/trocr-small-handwritten")
    ds = NHMDDataset("../unilm/trocr/NHMD_GT", "test", processor, max_length, augment=False)
    dataloader = DataLoader(ds, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("./nhmd_ocr/nhmd_out/nhmd_base_roberta_271k_e", "./nhmd_ocr/nhmd_out/nhmd_base_roberta_271k_d")
#    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("./nhmd_out/nhmd_small_full_271k_e", "./nhmd_out/nhmd_small_full_271k_d")
   # model = VisionEncoderDecoderModel.from_pretrained("./nhmd_ocr/nhmd_out/small_full/nhmd_small_full_final")
#    model = VisionEncoderDecoderModel.from_pretrained("./nhmd_ocr/nhmd_out/base_full/checkpoint-160000")
#    model = AutoModel.from_pretrained("./nhmd_ocr/nhmd_out/base_roberta/nhmd_base_roberta_final")
    

    # ROBERTA FRESH
    config = VisionEncoderDecoderConfig.from_pretrained("./nhmd_ocr/nhmd_out/base_roberta/nhmd_base_roberta_final")
    config.decoder_start_token_id = processor.tokenizer.cls_token_id
    config.pad_token_id = processor.tokenizer.pad_token_id
    # # make sure vocab size is set correctly

    # # set beam search parameters
    config.eos_token_id = processor.tokenizer.sep_token_id
    # config.max_length = max_length
    # config.early_stopping = True
    # config.no_repeat_ngram_size = 3
    # config.length_penalty = 2.0
    # config.num_beams = 4
    model = VisionEncoderDecoderModel._from_config(config)
    # model.config.vocab_size = model.config.decoder.vocab_size
    
    model.to(device)

    cer = load_metric('cer')
    wer = load_metric('wer')

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
         wer.add_batch(predictions=[s.lower() for s in pred_str], references=[s.lower() for s in label_str])

         preds += f"{pred_str}\t{label_str}\n"

    cer_final_score = cer.compute()
    wer_final_score = wer.compute()
    print(cer_final_score)
    print(wer_final_score)
    preds += f"cer: {str(cer_final_score)}\n"
    preds += f"wer: {str(wer_final_score)}\n"
#    with open("results_nhmd_base_roberta_final.txt", "w") as f:
#         f.write(preds)

if __name__ == '__main__':
    run_evaluate()
