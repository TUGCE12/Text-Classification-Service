import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from tqdm import tqdm
pretrained_model_name = 'bert-base-multilingual-cased'
from DataBaseManagements import getDataFromDB as gddb
from DataBaseManagements import getDataFromDB



def define_train_parameters(model, train_loader):
    """
    Define parameters optimize and, scheduler for using train step
    :param model: BertForSequenceClassification.from_pretrained model object
    :param train_loader: torch.utils.data.DataLoader object, which is created from your datas
    :return: optimizer, scheduler
    """
    # Define training parameters
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*5)
    return optimizer, scheduler


def train_step(model, train_loader, optimizer, scheduler, num_epochs=1):
    """
    Train your model
    :param model: BertForSequenceClassification.from_pretrained model object
    :param train_loader: torch.utils.data.DataLoader object, which is created from your datas
    :param optimizer: torch.optim.AdamW optimizer was used
    :param scheduler: transformers.get_linear_schedule_with_warmup was used
    :param num_epochs: default = 1
    :return: model and loss values: model, loss
    """
    epoch_loss=0
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'labels': batch['labels']
                }
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                pbar.update(1)

                pbar.set_postfix({'Loss': epoch_loss / (pbar.n + 1)})  # Update the loss in the progress bar
            pbar.close()
    return model, epoch_loss/(pbar.n + 1)


def eval_step(model, test_loader, test_data, label_encoder):
    """
    Evaluate your model
    :param model: BertForSequenceClassification.from_pretrained model object
    :param test_loader: torch.utils.data.DataLoader object, which is created from your datas
    :param test_data: pandas dataframe, will use calculate accuracy and classification_report
    :param label_encoder: sklearn.preprocessing, LabelEncoder object
    :return: accuracy, classify_report
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels']
            }
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).tolist()
            labels = batch['labels'].tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Convert predictions back to original labels
    predicted_labels = label_encoder.inverse_transform(all_preds)

    # Calculate accuracy and classification report
    accuracy = accuracy_score(test_data['label'], predicted_labels)  # accuracy_score = from scikit learn
    classify_report = classification_report(test_data['label'], predicted_labels)

    return accuracy, classify_report

