from fastapi import File, UploadFile, HTTPException
from DataBaseManagements import saveToDB, getDataFromDB
import pandas as pd
from typing import Union
from fastapi import FastAPI

app = FastAPI()
database_file = "/Users/tugcecelik/Desktop/Final_Project/TextClassifierService/classify_service.db"

@app.post("/create_model/")
async def create_model(model_name: str, model_service: str):
    """
    Endpoint for create a model and return model_id
    """
    model_id = saveToDB.save_model_to_database(database_file=database_file, model_name=model_name, model_service=model_service)
    return {"status": "success", "model_id": model_id}

@app.post("/upload_file/")
async def upload_csv_file(model_id: int, file: UploadFile = File(...)):
    """
    Endpoint for uploading a CSV file and saving data to the database.
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Read CSV file into a pandas dataframe
    try:
        df = pd.read_csv(file.file)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")

    code, message = saveToDB.save_data_to_database(database_file=database_file, model_id=model_id, pandas_dataframe=df)
    if code == 400:
        # Raise a 400 Bad Request error with a custom detail message
        raise HTTPException(status_code=400, detail=message)
    elif code == 200:
        raise HTTPException(status_code=200, detail=message)



@app.get("/get_all_data/")
async def get_all_data(model_id: int):
    """
    Endpoint for get model data
    """
    model_data = getDataFromDB.get_all_data_from_database_pd(database_file=database_file, model_id=model_id)
    return {"model_data": model_data.to_dict(orient="records")}