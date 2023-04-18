
from transformers import TrOCRProcessor, AutoFeatureExtractor, AutoTokenizer

def get_processor():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#    processor.image_processor.size = {'height': 64, 'width': 560}
    processor.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return processor
