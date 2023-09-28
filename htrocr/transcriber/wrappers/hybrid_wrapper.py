from htrocr.nhmd_hybrid.hybridnet import CNNTransformerHybrid
from htrocr.transcriber.nhmd_transcriber import Transcriber
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


class HybridTranscriber(Transcriber):
    def __init__(self, path):
        super().__init__(path)
        self.model = CNNTransformerHybrid.load_from_checkpoint(self.config['transcription_model_weight_path'])
        self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def transcribe(self, img):
        img = Image.fromarray(img).convert("L")
        w, h = img.size
        aspect_ratio = self.height / h
        new_width = round(w * aspect_ratio)
        img = img.resize((new_width, self.height))
        if new_width < 32:
            img = self.expand_img(img, self.height, 32)
        image = np.array(img)
        attn_images = []
        attn_images.append([1] * w)
        attn_images = (
            self.pool(self.pool(torch.tensor(attn_images).float().to(self.device))).long()
        )
        to_tensor = ToTensor()
        images = [to_tensor(image).to(self.device)]
        pred = self.model(images, attn_images)
        return self.decode(pred)
