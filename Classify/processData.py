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


# Create PyTorch datasets

class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokens['input_ids'][idx],
            'attention_mask': self.tokens['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_dataset_from_db(database_file, model_id):
    """
    Load your model dataset and drop id column for ai model
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :return: pandas dataframe, dataset
    """
    # Load data as pandas dataframe
    dataset_pd = getDataFromDB.get_all_data_from_database_pd(database_file, model_id)
    print(dataset_pd.head())
    # delete id column
    dataset_pd = dataset_pd.drop(columns=['id'])
    print(dataset_pd.head())
    return dataset_pd


def load_model_tokenizer_encoder(pretrained_model_name: str, dataset):
    """
    Load model, tokenizer and label_encoder information.
    :param pretrained_model_name: transformers models names: for classify it is 'bert-base-multilingual-cased'
    :param dataset: pandas dataframe, columns: text, label
    :return: model, tokenizer, label_encoder
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name,
                                                          num_labels=len(set(dataset['label'])))
    label_encoder = LabelEncoder()
    return model, tokenizer, label_encoder


def process_data(dataset, tokenizer, label_encoder, batch_size=8, test_size=0.2, random_state=42):
    """
    Prepare dataset for train end evaluation step.
    :param dataset: pandas dataframe, columns: text, label
    :param tokenizer: BertTokenizer.from_pretrained object
    :param label_encoder: sklearn.preprocessing, LabelEncoder object
    :param batch_size: default 8
    :param test_size: default 2
    :param random_state: default 42
    :return: train_loader, test_loader, test_data
    """
    # Check if the DataFrame is empty
    if dataset.empty:
        raise ValueError("DataFrame is empty. Please upload your data first!")

    else:
        # Split the dataset into training and testing sets
        train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=random_state)

        # Instantiate and fit the label encoder
        train_labels = label_encoder.fit_transform(train_data['label'])
        test_labels = label_encoder.transform(test_data['label'])

        # Tokenize and encode the text data
        train_tokens = tokenizer.batch_encode_plus(
            train_data['text'].tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        test_tokens = tokenizer.batch_encode_plus(
            test_data['text'].tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        train_dataset = CustomDataset(train_tokens, train_labels)
        test_dataset = CustomDataset(test_tokens, test_labels)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, test_data



