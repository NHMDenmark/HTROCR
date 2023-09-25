from abc import ABC, abstractmethod
from htrocr.nhmd_hybrid.nhmdtokenizer import NHMDTokenizer
import torch
from itertools import groupby
import json

from skimage.color import rgb2gray

class Transcriber(ABC):
    def __init__(self, path):
        self.tokenizer = NHMDTokenizer()
        with open(path) as f:
            self.config = json.load(f)

    def decode(self, pred):
        pred = pred.permute(1, 0, 2)
        logits = pred.log_softmax(2)
        _, max_index = torch.max(logits, dim=2)
        raw_prediction = list(max_index[:, 0].detach().cpu().numpy())
        prediction_idxs = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != self.tokenizer.pad_token_id])
        try:
            # try to stop at eos_token_id for a specific line in batch
            idx = prediction_idxs.index(1)
            decoded_text = prediction_idxs[: idx + 1]
        except ValueError:
            decoded_text = prediction_idxs
        pred_str = self.tokenizer.decode(decoded_text, skip_special_tokens=True)
        return str(pred_str)

    @abstractmethod
    def transcribe(self, img):
        pass
