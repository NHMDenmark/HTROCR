from transformers import TrOCRProcessor, AutoFeatureExtractor, AutoTokenizer


def get_processor(encoder_name, decoder_name):
    image_processor = AutoFeatureExtractor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessor(image_processor, tokenizer)
    #processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    return processor
