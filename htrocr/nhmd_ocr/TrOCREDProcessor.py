
from transformers import TrOCRProcessor, AutoTokenizer

def get_processor(processor, decoder=None):
    processor = TrOCRProcessor.from_pretrained(processor)
    if decoder is not None:
        processor.tokenizer = AutoTokenizer.from_pretrained(decoder)
    return processor
