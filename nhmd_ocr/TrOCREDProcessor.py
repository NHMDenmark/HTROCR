
from transformers import TrOCRProcessor, AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

def get_processor(processor, decoder=None):
    processor = TrOCRProcessor.from_pretrained(processor)
#    processor.image_processor.size = {'height': 64, 'width': 560}
    if decoder is not None:
        processor.tokenizer = AutoTokenizer.from_pretrained(decoder)
    return processor


#print()
#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#print(processor)
#model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("./out/nhmd_small_e", "./out/nhmd_small_d")
#print(model.encoder.config.hidden_size)
#print(model.decoder.config.hidden_size)
#print(model.config)
