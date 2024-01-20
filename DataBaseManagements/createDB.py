import sqlite3
import os
"""
This file created for setup base database
Run just one time for each client.

"""
# Specify the path to your SQLite database file
database_file = 'classify_service.db'

# Check if the file exists before attempting to delete it
if os.path.exists(database_file):
    os.remove(database_file)
    print(f"The database '{database_file}' has been deleted.")
else:
    print(f"The database file '{database_file}' does not exist.")


# Connect to SQLite database
conn = sqlite3.connect('classify_service.db')
cursor = conn.cursor()

# Create MODELS table (for bert models it is work)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS MODELS (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        model_service TEXT,
        saved_model_path TEXT,
        saved_tokenizer_path TEXT,
        saved_label_encoder_path TEXT,
        dataset_table_name TEXT,
        model_results_table_name TEXT
    )
''')

# Commit changes and close connection
conn.commit()
conn.close()

print("Database and MODELS table was created")
