import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
ACCUMULATION = 2
BERT_MODEL = 'bert-base-uncased'
MODEL_PATH = 'model.bin'
TRAIN_FILE = '~/Data/IMDB/IMDB_dataset.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_MODEL, 
                do_lower_case=True)

