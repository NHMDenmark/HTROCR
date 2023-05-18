from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from nhmd_hybrid.nhmdtokenizer import NHMDTokenizer
from transcriber.nhmd_transcriber import Transcriber
from PIL import Image
import numpy as np
import torch
import albumentations as A

class VisionEncoderDecoderTranscriber(Transcriber):
    def __init__(self, path='./transcriber/config.json'):
        super().__init__(path)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config['model']).to(self.device)
        self.processor = TrOCRProcessor.from_pretrained(self.config['processor'])
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
