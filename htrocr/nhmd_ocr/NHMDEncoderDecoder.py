from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from htrocr.nhmd_ocr.TrOCREDProcessor import get_processor

def fine_tune_model(encoder, decoder, base_arch, tokenizer_name):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)

    processor = get_processor(base_arch, tokenizer_name)
    
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 300
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    return model, processor

def generate_model(encoder, decoder_name, processor_name, max_length):
    "Load from encoder and decoder separately"
    decoder_config = AutoConfig.from_pretrained(decoder_name)
    decoder_config.max_length = max_length
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_config)
    
    model = VisionEncoderDecoderModel.from_pretrained(encoder)
    model.decoder = decoder
    
    processor = get_processor(processor_name, decoder_name)
    
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

def generate_model_full(model_name, processor_name, max_length):
    "Load full config"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    processor = get_processor(processor_name)

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
