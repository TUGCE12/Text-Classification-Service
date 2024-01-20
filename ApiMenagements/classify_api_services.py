from fastapi import File, UploadFile, HTTPException
from DataBaseManagements import saveToDB, getDataFromDB, deleteFromDB
import pandas as pd
from typing import Union
from fastapi import FastAPI
from Classify import classify, saveModelAndResults
app = FastAPI()
database_file = "/Users/tugcecelik/Desktop/Final_Project/TextClassifierService/classify_service.db"


@app.post("/create_model/")
async def create_model(model_name: str, model_service: str):
    """
    Endpoint for create a model and return model_id
    """
    code, model_id = saveToDB.save_model_to_database(database_file=database_file, model_name=model_name, model_service=model_service)
    if code == 200:
        return {"status": "success", "model_id": model_id}
    if code == 400:
        raise HTTPException(status_code=400, detail=model_id)


@app.get("/get_all_models/")
async def get_all_models():
    model_df = getDataFromDB.get_all_model_from_database_pd(database_file=database_file)
    return (model_df.to_dict(orient="records"))


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
    model_check = getDataFromDB.get_model_from_database(database_file=database_file, model_id=model_id)
    if model_check is None:
        raise HTTPException(status_code=400, detail=f"Model not found or did not exist at all")
    else:
        model_data = getDataFromDB.get_all_data_from_database_pd(database_file=database_file, model_id=model_id)
        return {"model_data": model_data.to_dict(orient="records")}


@app.post("/train/")
async def train(model_id: int, pretrained_model_name: str = "bert", over_train: bool = False):
    code, message = classify.classify_train_for_api(database_file=database_file,
                                                    model_id=model_id,
                                                    pretrained_model_name=pretrained_model_name,
                                                    use_old_knowledge= over_train)
    raise HTTPException(status_code=code, detail=message)


@app.get("/get_model_metrics/")
async def get_model_metrics(model_id:int):
    model_check = getDataFromDB.get_model_from_database(database_file=database_file, model_id=model_id)
    if model_check is None:
        raise HTTPException(status_code=400, detail=f"Model not found or did not exist at all")
    else:
        results = getDataFromDB.get_model_results_from_database(database_file=database_file, model_id=model_id)
        return {"model_metrics": results.to_dict(orient="records")}


@app.post("/prediction/")
async def make_predicion(model_id: int, text: str):
    model_check = getDataFromDB.get_model_from_database(database_file=database_file, model_id=model_id)
    if model_check is None:
        raise HTTPException(status_code=400, detail=f"Model not found or did not exist at all")
    else:
        is_model_saved = saveModelAndResults.check_model_saved_or_not(database_file=database_file, model_id=model_id)
        if is_model_saved:
            prediction = classify.make_single_prediction(database_file=database_file, model_id=model_id, input_text=text)
            return {"prediction": prediction}
        else:
            raise HTTPException(status_code=400, detail="For make prediction: Train your model!")


@app.delete("/delete_data/")
async def delete_data(model_id:int, data_id: int):
    model_check = getDataFromDB.get_model_from_database(database_file=database_file, model_id=model_id)
    if model_check is None:
        raise HTTPException(status_code=400, detail=f"Model not found or did not exist at all")
    else:

        check, code, message = deleteFromDB.delete_data_from_database(database_file=database_file,
                                                                      model_id=model_id,
                                                                      data_id=data_id)
        raise HTTPException(status_code=code, detail=message)

@app.delete("/delete_all_data/")
async def delete_data(model_id: int):
    model_check = getDataFromDB.get_model_from_database(database_file=database_file, model_id=model_id)
    if model_check is None:
        raise HTTPException(status_code=400, detail=f"Model not found or did not exist at all")
    else:

        check, code, message = deleteFromDB.delete_all_data_from_database(database_file=database_file,
                                                                          model_id=model_id)
        raise HTTPException(status_code=code, detail=message)

