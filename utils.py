import os

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for index, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            predictions.extend(preds)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), predictions, real_values


def save_model(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model checkpoint to %s", output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.nlp.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(model.out.state_dict(), output_dir)


def get_predictions(model, data_loader, device):
    model = model.eval()
    texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            text = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            texts.extend(text)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.show()


def show_length_graph(tokenizer, df):
    token_lens = []
    for txt in df.text:
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))
    sns.distplot(token_lens)
    plt.xlim([0, 256])
    plt.xlabel('Token count')
    plt.show()


def remove_non_ascii_chars(text):
    """
    return text after removing non-ascii characters i.e. characters with ascii value >= 128
    """
    return ''.join([w if ord(w) < 128 else ' ' for w in text])
