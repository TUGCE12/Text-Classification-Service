import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from tqdm import tqdm
from DataBaseManagements import getDataFromDB as gddb
from DataBaseManagements import getDataFromDB
from transformers import BertConfig
import joblib
import os

pretrained_model_name = 'bert-base-multilingual-cased'


def save_model(database_file, model_id, model, tokenizer, label_encoder):
    """
    Save your trained model for using later.
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :param model: BertForSequenceClassification.from_pretrained model object
    :param tokenizer: BertTokenizer.from_pretrained object
    :param label_encoder: sklearn.preprocessing, LabelEncoder object
    :return: no return
    """
    model_save_path, tokenizer_save_path, label_encoder_save_path = gddb.get_model_saved_paths(
        database_file=database_file,
        model_id=model_id)
    # Save model
    model.save_pretrained(model_save_path)

    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_save_path)

    # Save label encoder
    os.makedirs(label_encoder_save_path, exist_ok=True)
    joblib.dump(label_encoder, f'{label_encoder_save_path}/encoder')
    print("Model Successfully Saved\n")


def check_model_saved_or_not(database_file, model_id):
    """
    Check your model has a trained version or not
    :param database_file:
    :param model_id:
    :return:  true model saved, false model not saved
    """
    model_path, tokenizer_path, label_encoder_path = getDataFromDB.get_model_saved_paths(database_file=database_file,
                                                                                         model_id=model_id)

    check_paths = os.path.exists(model_path) and os.path.isdir(model_path) \
                  and os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path) \
                  and os.path.exists(label_encoder_path) and os.path.isdir(label_encoder_path)

    # check path = true all path exist model may was saved, false model not saved ever
    if check_paths:
        m_p = os.listdir(model_path)
        t_p = os.listdir(tokenizer_path)
        l_e_p = os.listdir(label_encoder_path)
        if len(m_p) == 0 and len(t_p) == 0 and len(l_e_p) == 0:  # if true all is empty or at least one of them empty
            # load standart way
            return False
        else:
            # load saved model
            return True
    else:
        # load standart way
        return False

