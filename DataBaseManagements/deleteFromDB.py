import sqlite3
database_file = 'classify_service.db'


def delete_data_from_database(database_file, model_id, data_id):
    """
    Delete a data from dataset
    :param database_file: sqlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :param data_id: int, unique id for your data in your dataset
    :return: True: if data deleted, False: if data not found
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()
    table_name = row[6]
    # Execute the DELETE statement
    cursor.execute(f'DELETE FROM {table_name} WHERE id = ?', (data_id,))
    # Check if any rows were affected

    if cursor.rowcount > 0:
        message = f"Deletion successful. {cursor.rowcount} row(s) deleted."
        # Commit the changes
        conn.commit()
        conn.close()
        return True, 200, message
    else:
        message = f"No rows deleted. ID {data_id} not found."
        # Commit the changes
        conn.commit()
        conn.close()
        return False, 400, message


def delete_all_data_from_database(database_file, model_id):
    """
    Delete all data from your dataset table
    :param database_file: qlite database file path
    :param model_id: int, unique id for your model, can find in MODELS table
    :return: True: if data deleted, False: if data not found
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # get model information for find model dataset table
    cursor.execute('SELECT * FROM MODELS WHERE model_id = ?', (model_id,))
    row = cursor.fetchone()
    table_name = row[6]
    # Execute the DELETE statement
    cursor.execute(f'DELETE FROM {table_name}')
    # Check if any rows were affected

    if cursor.rowcount > 0:
        message = f"Deletion successful. {cursor.rowcount} row(s) deleted."
        # Commit the changes
        conn.commit()
        conn.close()
        return True, 200, message
    else:
        # Commit the changes
        conn.commit()
        conn.close()
        message = f"No rows deleted."
        return False, 400, message
