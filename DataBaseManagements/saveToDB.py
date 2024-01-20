import sqlite3
database_file = 'classify_service.db'


def save_model_to_database(database_file, model_name, model_service):
    """
    this function add a row into your database MODELS table, and create model information,
    model dataset table and model results table.
    :param database_file: sqlite database file path
    :param model_name: You can give your model any name you want.
                        There might be something consistent with the data
                        that you can remember when you look at it later.
    :param model_service: for now we have just "classify" service
    :return: no return
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Generate dataset_table_name
    cursor.execute('SELECT COALESCE(MAX(model_id), 0) + 1 FROM MODELS')
    model_id = cursor.fetchone()[0]
    dataset_table_name = f"{model_service}_{str(model_id)}"
    saved_model_path = f"models/model_{dataset_table_name}"
    saved_tokenizer_path = f"tokenizers/tokenizer_{dataset_table_name}"
    saved_label_encoder_path = f"encoders/encoder_{dataset_table_name}"
    model_results_table_name = f"results_{model_service}_{str(model_id)}"

    # Insert data into MODELS table
    cursor.execute('''
        INSERT INTO MODELS (model_name, model_service, saved_model_path, saved_tokenizer_path, saved_label_encoder_path, dataset_table_name, model_results_table_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (model_name, model_service, saved_model_path, saved_tokenizer_path, saved_label_encoder_path, dataset_table_name, model_results_table_name))

    # Create dataset table
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {dataset_table_name} (
            id INTEGER,
            text TEXT,
            label TEXT
        )
    ''')

    # Create model results table
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {model_results_table_name} (
            loss REAL,
            accuracy REAL,
            classification_report TEXT
        )
    ''')
    # Commit changes and close connection
    conn.commit()
    conn.close()
    return model_id


def save_data_to_database(database_file, model_id, pandas_dataframe):
    """
    Save your data in your model database table in DataBase.
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :param pandas_dataframe: working dataset as pandas dataframe format.
                                for classify: text, label, id columns are required.
    :return: no return
    """
    # Check the number of columns
    column_count = pandas_dataframe.shape[1]
    column_names = pandas_dataframe.columns.tolist()
    if 'label' in column_names:
        if 'text' in column_names:
            if 'id' in column_names:
                pass
            else:
                pandas_dataframe['id'] = pandas_dataframe.reset_index().index
        else:
            message = """The data columns you are trying to upload is not appropriate, Your data must have text, label and id columns."""
            return 400, message
    else:
        message = """The data columns you are trying to upload is not appropriate, Your data must have text, label and id columns."""
        return 400, message



    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    # Save the DataFrame to the DATASET table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()

    # Check if the row is found
    if row:
        pandas_dataframe.to_sql(row[6], conn, index=False, if_exists='replace')
        # Set if_exists to 'replace' to overwrite the table if it exists
        message = "model_id found: Data added successfully"
        print(message)
        conn.close()
        return 200, message

    else:
        message = f"model_id = {model_id} is not found."
        print(message)
        conn.close()
        return 400, message



def save_model_results_to_database(database_file, model_id, loss, accuracy, classify_report):
    """
    Save model results into model results table
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :param loss: model train loss value after train step
    :param accuracy: model accuracy after evaluation step
    :param classify_report: model classify report after evaluation step
    :return: no return
    """
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    # Save the DataFrame to the DATASET table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()

    # Check if the row is found
    if row:
        table_name = row[7]
        # Insert the results into the table
        cursor.execute(f'''
            INSERT INTO {table_name} (loss, accuracy, classification_report)
            VALUES (?, ?, ?)
        ''', (loss, accuracy, classify_report))
        print("Results Successfully Saved")

    else:
        print(f"model_id = {model_id} is not found.")

    conn.commit()
    conn.close()
