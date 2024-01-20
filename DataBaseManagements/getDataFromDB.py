import sqlite3
database_file = 'classify_service.db'
import pandas as pd


def get_all_data_from_database_list(database_file, model_id):
    """
    Get all data from your model as list format
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :return: list, all data from your model
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()

    # from model dataset table get all data
    cursor.execute(f'SELECT * FROM {row[6]}')
    dataset = cursor.fetchall()
    conn.close()
    return dataset


def get_all_data_from_database_pd(database_file, model_id):
    """
    Get all data from your model as pandas dataframe format
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :return: pandas dataframe, all data from your model
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()

    # Query to select all data from the specified table
    query = f'SELECT * FROM {row[6]}'

    # Read data from the database into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_data_from_database(database_file, model_id, data_id):
    """
    get a specific data with data id from your model
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :param data_id: int, unique id for your data in your dataset
    :return: list, empty if data was not exist: all row if data exist
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()

    cursor.execute(f'SELECT * FROM {row[6]} WHERE id = ?', (data_id,))
    data = cursor.fetchall()
    conn.close()
    return data


def get_all_model_from_database_list(database_file):
    """
    Get models list
    :param database_file: sqlite database file path
    :return: list, all model list
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS')
    models = cursor.fetchall()
    conn.close()
    return models


def get_all_model_from_database_pd(database_file):
    """
    Get all models as pandas dataframe format
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :return: pandas dataframe, all model
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)

    # Read data from the database into a Pandas DataFrame
    model_df = pd.read_sql_query('SELECT * FROM MODELS', conn)
    conn.close()
    return model_df


def get_model_from_database(database_file, model_id):
    """
    get a model info
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :return: list, empty: if model was not exist, model information: if model exist
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    model_info = cursor.fetchone()

    return model_info


def get_model_saved_paths(database_file, model_id):
    """
    Gets the directories where the model, tokenizer and label_encoder will be saved
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :return: three paths: model_path, tokenizer_path and label_encoder_path
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    model_info = cursor.fetchone()
    # encoder path index is 5, tokenizer path index is 4, and model path index is 3

    return model_info[3], model_info[4], model_info[5]


def get_model_results_from_database(database_file, model_id):
    """
    this function return a model's all results
    :param database_file: sqlite database file path
    :param model_id:  int, unique id for your model, can find in MODELS table
    :return: models results
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()

    # Query to select all data from the specified table
    query = f'SELECT * FROM {row[7]}'

    # Read data from the database into a Pandas DataFrame
    results = pd.read_sql_query(query, conn)
    conn.close()
    return results

