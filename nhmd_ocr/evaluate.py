from nhmd_ocr.NHMDDataset import NHMDDataset
from nhmd_ocr.TrOCREDProcessor import get_processor
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
import torch
from tqdm import tqdm
from datasets import load_metric
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def run_evaluate():
    max_length = 300

    # Load TrOCR Base image processor and a tokenizer from xlm-roberta-base decoder.
    processor = get_processor("microsoft/trocr-base-handwritten", 'xlm-roberta-base')

    # Load image processor and tokenizer from the same TrOCR model
#    processor = get_processor("microsoft/trocr-base-handwritten")
#    processor = get_processor("microsoft/trocr-small-handwritten")
    
    # Specify path to ground truth
    ds = NHMDDataset("../unilm/trocr/NHMD_GT", "test", processor, max_length, augment=False)
    dataloader = DataLoader(ds, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
#    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("./nhmd_ocr/nhmd_out/nhmd_base_roberta_271k_e", "./nhmd_ocr/nhmd_out/nhmd_base_roberta_271k_d")
#    model = VisionEncoderDecoderModel.from_pretrained("./nhmd_ocr/nhmd_out/small_full/nhmd_small_full_final")
    model = VisionEncoderDecoderModel.from_pretrained("./nhmd_ocr/nhmd_out/base_full/nhmd_base_full_final")
#    model = AutoModel.from_pretrained("./nhmd_ocr/nhmd_out/base_roberta/nhmd_base_roberta_final")

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
    with open("results_nhmd_base_full_final.txt", "w") as f:
        f.write(preds)

if __name__ == '__main__':
    run_evaluate()
