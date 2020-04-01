import transformers

MAX_LEN = 192
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 4
EPOCHS = 5
BERT_PATH = "../input/bert-base-multilingual-uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE1 = "../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv"
TRAINING_FILE2 = "../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv"
VALIDATION_FILE = "../input/jigsaw-multilingual-toxic-comment-classification/validation.csv"
TEST_FILE = "../input/jigsaw-multilingual-toxic-comment-classification/test.csv"

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)