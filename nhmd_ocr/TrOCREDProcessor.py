from transformers import TrOCRProcessor, AutoFeatureExtractor, AutoTokenizer


def get_processor(encoder_name, decoder_name):
#    image_processor = AutoFeatureExtractor.from_pretrained(encoder_name,size=(64,560))
#    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
#    processor = TrOCRProcessor(image_processor, tokenizer)
#    image_processor.image_size = (32, 640)
#    model.config.image_size = (256, 256)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    processor.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return processor
