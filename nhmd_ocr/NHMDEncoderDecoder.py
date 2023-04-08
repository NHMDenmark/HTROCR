from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from TrOCREDProcessor import get_processor

def fine_tune_model(decoder_name):
    model = VisionEncoderDecoderModel.from_pretrained("out/checkpoint-50000")
    processor = TrOCRProcessor.from_pretrained("out/checkpoint-50000")
    return model, processor

def generate_model(encoder_name, decoder_name, max_length, num_decoder_layers=None):
#    encoder_config = AutoConfig.from_pretrained(encoder_name)
#    encoder_config.is_decoder = False
#    encoder_config.add_cross_attention = False
#    encoder = AutoModel.from_config(encoder_config)

    decoder_config = AutoConfig.from_pretrained(decoder_name)
    decoder_config.max_length = max_length
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_config)
    
#    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
#    config.tie_word_embeddings = False
#    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

#    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.decoder = decoder

    processor = get_processor(encoder_name, decoder_name)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, processor
