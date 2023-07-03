from nhmd_vit.VitCTC import ViTCTC
from transcriber.nhmd_transcriber import Transcriber
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import torch


class VitTranscriber(Transcriber):
    def __init__(self, path):
        super().__init__(path)
        self.model = ViTCTC.load_from_checkpoint(self.config['transcription_model_weight_path'])
        self.maxlen = 100
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def transcribe(self, img):
        img = Image.fromarray(img).convert("L")
        img = img.resize((384, 384))
        image = np.array(img)
        to_tensor = ToTensor()
        images = [to_tensor(image).to(self.device)]
        pred = self.model(images, self.maxlen)
        return self.decode(pred)