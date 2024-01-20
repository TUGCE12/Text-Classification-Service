from DataBaseManagements import saveToDB as sm
from DataBaseManagements import getDataFromDB as gd
from DataBaseManagements import deleteFromDB as ddb
import os
import pandas as pd
import sqlite3

if os.path.exists(sm.database_file):
    print(f"The database '{sm.database_file}' exist.\n")
else:
    print(f"The database file '{sm.database_file}' does not exist.")
    from DataBaseManagements import createDB

# from DataBaseManagements import createDB




data = pd.read_csv('labeled-haber.csv')
data = data.rename({'category': 'label', 'text': 'text'}, axis='columns')
data['id'] = data.reset_index().index

# Print the updated DataFrame
print(data.head())



# Connect to the SQLite database
conn = sqlite3.connect(sm.database_file)
cursor = conn.cursor()

# Define the model_id you want to retrieve
desired_model_id = 3


print("\n\n Yeni fonk denemesi")
sm.save_data_to_database(sm.database_file, 1, data)
dataset = gd.get_all_data_from_database_list(sm.database_file, 1)
print(type(dataset))
i = 0
for data_r in dataset:
    print(data_r)
    if i ==10:
        break
    i = i+1

dataset = gd.get_all_data_from_database_pd(sm.database_file, 1)
print(dataset.head())


# DELETE
# Execute the SELECT statement
ddb.delete_data_from_database(sm.database_file, 1, 1)
d = gd.get_data_from_database(sm.database_file, 1, 1)
print(d)
d = gd.get_data_from_database(sm.database_file, 1, 2)
print(d)
# ddb.delete_data_from_database(sm.database_file, 1, 2)
# d = gd.get_data_from_database(sm.database_file, 1, 2)
# print(d)
# ddb.delete_data_from_database(sm.database_file, 1, 2)
# d = gd.get_data_from_database(sm.database_file, 1, 2)
# print(d)
#
# ddb.delete_all_data_from_database(sm.database_file, 1)
# d = gd.get_all_data_from_database_list(sm.database_file, 1)
# print(d)
d = gd.get_all_data_from_database_pd(sm.database_file, 1)
print(d)

m = gd.get_all_model_from_database_list(sm.database_file)
print(m)
m = gd.get_all_model_from_database_pd(sm.database_file)
print(m)



# sm.save_model_to_database(sm.database_file, 'your_model_name', 'classify')
# sm.save_model_to_database(sm.database_file, 'your_model_name2', 'classify')
#
#
# # Connect to the SQLite database
# conn = sqlite3.connect(sm.database_file)
# cursor = conn.cursor()
#
#
# # Example data to be inserted
# example_data = [
#     (1, 'Text 1', 'Label 1'),
#     (2, 'Text 2', 'Label 2'),
#     # Add more rows as needed
# ]
#
# # Fetch all rows from the MODELS table
# cursor.execute('SELECT dataset_table_name FROM MODELS')
# rows = cursor.fetchall()
#
# # Print the retrieved data
# for row in rows:
#     print(row)
#
#     # Insert the data into the DATASET table
#     cursor.executemany(f'INSERT INTO {row[0]} (id, text, label) VALUES (?, ?,?)', example_data)
#
#     cursor.execute(f'SELECT * FROM {row[0]}')
#     datas = cursor.fetchall()
#     for data in datas:
#         print(data)
#
#
# # Fetch all rows from the MODELS table
# cursor.execute('SELECT * FROM MODELS')
# rows = cursor.fetchall()
#
# # Print the retrieved data
# for row in rows:
#     print(row)
#     print(row[6])
#
#
# # Close the connection
# conn.close()

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from tqdm import tqdm
from Classify import processData as psd


print("\n#########################\nBaşla")
dataset = psd.load_dataset_from_db(sm.database_file, 1)
print(dataset)
# model, tokenizer, label_encoder = psd.load_model_tokenizer_encoder(psd.pretrained_model_name, dataset)
# print(model)
# print(tokenizer)
# print(label_encoder)
#
# train_loader , test_loader, test_data = psd.process_data(dataset=dataset, tokenizer=tokenizer, label_encoder=label_encoder)
# print(train_loader)
# print(test_loader)
# print(test_data)

# from Classify import trainEvalSteps as tes
# # Define training parameters
# optimizer, scheduler = tes.define_train_parameters(model=model, train_loader=train_loader)
#
#
# model, epoch_loss = tes.train_step(model=model, train_loader=train_loader, optimizer=optimizer,scheduler=scheduler)
# # print(model)
# print(epoch_loss)
#
# accuracy, class_report = tes.eval_step(model=model, test_loader=test_loader, test_data=test_data, label_encoder=label_encoder)
# print(accuracy)
# print(class_report)
#
# print(gd.get_model_saved_paths(sm.database_file, 1))
# # Save model performance results to DB
#
# # Save model for use later (save to path which define in db)
# from Classify import saveModelAndResults as saveMR
#
# saveMR.save_model(database_file=sm.database_file, model_id=1,model=model, tokenizer=tokenizer,label_encoder=label_encoder)
#
# sm.save_model_results_to_database(database_file=sm.database_file, model_id=1,loss=epoch_loss, accuracy=accuracy, classify_report=class_report)
# print(gd.get_model_results_from_database(database_file=sm.database_file, model_id=1))

#
# """get model tokenizer label encoder from saved path"""
# model, tokenizer, label_encoder = psd.load_model_tokenizer_encoder_from_saved_path(database_file=sm.database_file, model_id=1)
#
# train_loader, test_loader, test_data = psd.process_data(dataset=dataset, tokenizer=tokenizer, label_encoder=label_encoder)
# print(train_loader)
# print(test_loader)
# print(test_data)
#
# from Classify import trainEvalSteps as tes
# # Define training parameters
# optimizer, scheduler = tes.define_train_parameters(model=model, train_loader=train_loader)
#
#
# model, epoch_loss = tes.train_step(model=model, train_loader=train_loader, optimizer=optimizer,scheduler=scheduler)
# # print(model)
# print(epoch_loss)


from Classify import classify
code, message = classify.classify_train_for_api(database_file=sm.database_file, model_id=3, pretrained_model_name='bert')
print(code, message)
print(gd.get_model_results_from_database(database_file=sm.database_file, model_id=3))