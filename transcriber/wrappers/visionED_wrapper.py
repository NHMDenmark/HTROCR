from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from nhmd_hybrid.nhmdtokenizer import NHMDTokenizer
from transcriber.nhmd_transcriber import Transcriber
from PIL import Image
import numpy as np
import torch
import albumentations as A
from transformers.utils.logging import set_verbosity_error

class VisionEncoderDecoderTranscriber(Transcriber):
    def __init__(self, path):
        super().__init__(path)
        # Suppress image processor warnings - 
        # https://discuss.huggingface.co/t/error-finding-processors-image-class-loading-based-on-pattern-matching-with-feature-extractor/31890/6.
        set_verbosity_error()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.processor = TrOCRProcessor.from_pretrained(self.config['transcription_img_processor'])
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config['transcription_model_weight_path']).to(self.device)
        self.maxlen = 300

    def transcribe(self, img):
        img = Image.fromarray(img).convert("RGB")
        image = np.array(img)
        transform = A.ToGray(always_apply=True)
        img_transformed = transform(image=image)['image']
        pixel_values = self.processor(img_transformed, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze().to(self.device)
        pixel_values = torch.unsqueeze(pixel_values, dim=0)
        outputs = self.model.generate(pixel_values, max_length=self.maxlen)
        # decode
        pred_str = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return str(pred_str[0])
