import fire
import os
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, Trainer, TrainingArguments, TrainerCallback
from NHMDDataset import NHMDDataset
from NHMDEncoderDecoder import generate_model, generate_model_full, fine_tune_model
from MetricProcessor import MetricProcessor
import json
import time
import torch
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

class SaveCallback(TrainerCallback):
    def on_save_checkpoint(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            state.trainer.model.encoder.save_pretrained("./nhmd_out/nhmd_base_full_271k_e")
            state.trainer.model.decoder.save_pretrained("./nhmd_out/nhmd_base_full_271k_d")

def NHMD_collator(ins):
    thedict = default_data_collator(ins)
    thedict["interpolate_pos_encoding"] = True
    return thedict

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

def run(run_name=None):
    wandb.login()
    with open('config/default.json') as f:
        config = json.load(f)
    wandb.init(project="NHMD_OCR", config=config)

#    model, processor = generate_model("microsoft/trocr-large-handwritten", "xlm-roberta-base", "microsoft/trocr-large-handwritten", config['max_len'])
    model, processor = generate_model_full("microsoft/trocr-large-handwritten", "microsoft/trocr-large-handwritten", config['max_len'])
#    model, processor = generate_model_full("nhmd_out/small_full/checkpoint-40000", "microsoft/trocr-small-handwritten", config['max_len'])    
#    model, processor = fine_tune_model(config['decoder_name'])
    train_dataset = NHMDDataset(config['data_path'], "train", processor, config['max_len'], config['augment'])
    valid_dataset = NHMDDataset(config['data_path'], "valid", processor, config['max_len'], config['augment'])
    metrics = MetricProcessor(processor)


    if run_name == None:
        run_name = f'execution_no_{str(time.time())}'

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy=config['evaluation_strategy'],
        save_strategy=config['weight_save_strategy'],
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        fp16=config['fp16'],
        fp16_full_eval=config['fp16_eval'],
        dataloader_num_workers=config['dataloader_workers'],
        output_dir=config['output_dir'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        num_train_epochs=config['train_epochs'],
        run_name=run_name
    )

    trainer = Seq2SeqTrainer(
        model=model,
#        callbacks=[SaveCallback()],
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
#        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()


    model.save_pretrained("./nhmd_out/large_full/nhmd_large_full_final")

    wandb.finish()


if __name__ == '__main__':
    fire.Fire(run)
