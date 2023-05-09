from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, Trainer, TrainingArguments, TrainerCallback
from transformers import RobertaTokenizer,AutoTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
print('pad',tokenizer.pad_token_id)
print('sep',tokenizer.sep_token_id)
print('mask',tokenizer.mask_token_id)
print('cls',tokenizer.cls_token_id)
print('inp_ids', tokenizer("Test, Hello world")["input_ids"])
print(tokenizer)


