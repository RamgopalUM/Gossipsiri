from transformers import AdamW, get_cosine_schedule_with_warmup, BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import train_epoch, eval_model, show_confusion_matrix, get_predictions, show_length_graph

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
DATA_PATH = "data/twt_sample.csv"
DATA_PATH_2 = "data/stock_data.csv"
OUT_PATH = "output"
RANDOM_SEED = 46
MAX_LEN = 50
BATCH_SIZE = 32
EPOCHS = 8


def setup():
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    rcParams['figure.figsize'] = 12, 8
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


class TwitterDataset(Dataset):
    def __init__(self, text, targets, tokenizer, max_len):
        self.text = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=False):
    ds = TwitterDataset(
        text=df.text.to_numpy(),
        targets=df.sentiment_encode.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle
    )


def load_multiclass_data(tokenizer):
    bearish_data = pd.read_csv('training_data/bearish.txt', sep="\n", header=None)
    neutral_data = pd.read_csv('training_data/neutral.txt', sep="\n", header=None)
    bullish_data = pd.read_csv('training_data/bullish.txt', sep="\n", header=None, engine='python')

    bearish_data.columns = ["text"]
    neutral_data.columns = ["text"]
    bullish_data.columns = ["text"]

    bearish_data['sentiment_encode'] = [0] * bearish_data.shape[0]
    neutral_data['sentiment_encode'] = [1] * neutral_data.shape[0]
    bullish_data['sentiment_encode'] = [2] * bullish_data.shape[0]

    print(bearish_data.head(10), bearish_data.shape)
    print(neutral_data.head(10), neutral_data.shape)
    print(bullish_data.head(10), bullish_data.shape)
    df = pd.concat([bearish_data, neutral_data, bullish_data])
    print(df.shape, df.head(10))
    df = df.reset_index(drop=True)
    sns.countplot(df.sentiment_encode)
    plt.xlabel('Sentiment')
    plt.show()

    df_train, df_test = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=df['sentiment_encode']
    )
    df_train, df_val = train_test_split(
        df_train,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=df_train['sentiment_encode']
    )

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, True)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    return train_data_loader, val_data_loader, test_data_loader


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        #config = AlbertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME, classifier_dropout_prob=0.5, num_labels=n_classes)
        #self.nlp = AlbertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, config=config)
        #self.fc1 = nn.Linear(self.nlp.config.hidden_size, 128)
        self.nlp = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.4)
        self.out = nn.Linear(self.nlp.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.nlp(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def train():
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    #train_data_loader, val_data_loader, test_data_loader, class_weights = load_data(DATA_PATH, tokenizer)
    train_data_loader, val_data_loader, test_data_loader = load_multiclass_data(tokenizer)

    class_names = ['Bullish', 'Neutral', "Bearish"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SentimentClassifier(len(class_names))

    model = model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    #print(model.nlp.__repr__())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.nlp.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.nlp.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=8e-6, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    print(total_steps, len(train_data_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=20,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss, y_pred, y_test = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print('F1-micro: {}'.format(f1_score(y_test, y_pred, average='micro')))
        print('F1-macro: {}'.format(f1_score(y_test, y_pred, average='macro')))
        print('F1 2-classes: {}'.format(f1_score(y_test, y_pred, average=None)))
        #cm = confusion_matrix(y_test, y_pred)
        #df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        #show_confusion_matrix(df_cm)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_multi_class.bin')
            best_accuracy = val_acc

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()

    best_model = SentimentClassifier(len(class_names))
    best_model.load_state_dict(torch.load('best_model_multi_class.bin'))
    best_model.to(device)

    texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader,
        device
    )

    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)


def main():
    setup()
    train()


if __name__ == "__main__":
    main()
