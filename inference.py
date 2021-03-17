import getopt
import sys
from datetime import datetime
import json
from utils import get_predictions, show_confusion_matrix
from train_model import SentimentClassifier, create_data_loader
from sklearn.metrics import confusion_matrix, classification_report
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer

RANDOM_SEED = 46
MAX_LEN = 50
BATCH_SIZE = 32
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
DATA_PATH = "data/twt_sample.csv"


def load_stock_twits(file_path, tokenizer):
    if file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path)
        df.columns = ['created_at', 'text']
    elif file_path.split('.')[-1] == 'json':
        with open(file_path) as file:
            data = json.load(file)
            tweets = [[d['created_at'], d['body']] for d in data]
        print(len(tweets), tweets[0])
        df = pd.DataFrame(tweets, columns=['created_at', 'text'])
    else:
        print("Wrong input file type!")
        exit(1)
    # hardcode a sentiment
    df['sentiment_encode'] = -1
    df['text'] = df['text'].replace(r'http\S+', '', regex=True)
    keep_idx = []
    for i in df.index:
        date_time_obj = datetime.strptime(str(df['created_at'][i]), '%Y-%m-%d %H:%M:%S')
        if date_time_obj.weekday() < 5 and date_time_obj.day != 26:
            keep_idx.append(i)
    print(df.shape, df.iloc[keep_idx].shape)
    df = df.iloc[keep_idx]

    def clean_text(text):
        return text[2:-1]

    df['text'] = df.text.apply(clean_text)
    df = df[~df['text'].str.contains('RT')]

    #df['text'] = df['text'].replace(r'@\S+', '', regex=True)
    #df['text'] = df['text'].replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    #df['text'] = df['text'].replace(r'[$]\S+', '', regex=True)
    #count = df['text'].str.split().str.len()
    #df = df[~(count <= 2)]
    df = df.reset_index(drop=True)
    print(df.head(10))
    return df, create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)


def evaluate_test_result(y_pred, y_test):
    class_names = ['Bearish', 'Neutral', 'Bullish']
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)


def main(argv):
    input_file = ''
    output_file = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["input=", "output="])
    except getopt.GetoptError:
        print('inference.py -i <input_file> -o <output_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('inference.py -i <input_file> -o <output_file>')
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-o", "--output"):
            if arg.split('.')[-1] != 'csv':
                print("Wrong output file type!")
                exit(1)
            output_file = arg
    print('Input file is "', input_file)
    print('Output file is "', output_file)

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    df_original, data_loader = load_stock_twits(input_file, tokenizer)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = 'demo_bert_base_cased.bin'
    model = SentimentClassifier(3)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    #_, _, test_data_loader = load_multiclass_data(tokenizer)

    y_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        data_loader,
        device
    )

    #for a, b, c in zip(y_texts, y_pred, y_pred_probs):
    #    print(a, b, c)
    y_pred_df = pd.DataFrame(y_pred, columns=["pred_sentiment"])

    df = pd.concat([df_original, y_pred_df], axis=1, sort=False)
    df = df.drop(columns=['sentiment_encode'])
    print(df.head(10))
    df.to_csv(output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
