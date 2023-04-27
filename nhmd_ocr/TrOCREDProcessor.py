
from transformers import TrOCRProcessor, AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

def get_processor():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#    processor.image_processor.size = {'height': 64, 'width': 560}
    processor.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return processor

#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("./out/nhmd_small_e", "./out/nhmd_small_d")
print(model.encoder.config.hidden_size)
print(model.decoder.config.hidden_size)
print(model.decoder.config)
