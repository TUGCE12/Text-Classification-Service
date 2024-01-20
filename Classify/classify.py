from Classify import processData , trainEvalSteps, saveModelAndResults
from DataBaseManagements import getDataFromDB, saveToDB
import os
import torch


def classify_train_for_api(database_file, model_id, pretrained_model_name, use_old_knowledge: bool = False):
    # check model_id
    model_check = getDataFromDB.get_model_from_database(database_file=database_file, model_id=model_id)
    if model_check is None:
        return 400, f"Model not found or did not exist at all"

    # check model trained before or not
    is_model_saved = saveModelAndResults.check_model_saved_or_not(database_file=database_file, model_id=model_id)

    dataset = processData.load_dataset_from_db(database_file=database_file, model_id=model_id)
    if pretrained_model_name.lower() == "bert":
        pretrained_model_name = 'bert-base-multilingual-cased'

        if use_old_knowledge: # because may user don't want to use old model's bias and weight
            if is_model_saved:
                model, tokenizer, label_encoder = processData.load_model_tokenizer_encoder_from_saved_path(
                    database_file=database_file,
                    model_id=model_id)
            else:
                model, tokenizer, label_encoder = processData.load_model_tokenizer_encoder(
                    pretrained_model_name=pretrained_model_name,
                    dataset=dataset)
        else:
            model, tokenizer, label_encoder = processData.load_model_tokenizer_encoder(
                pretrained_model_name=pretrained_model_name,
                dataset=dataset)

        train_loader, test_loader, test_data = processData.process_data(dataset=dataset,
                                                                        tokenizer=tokenizer,
                                                                        label_encoder=label_encoder)

        optimizer, scheduler = trainEvalSteps.define_train_parameters(model=model, train_loader=train_loader)

        model, loss = trainEvalSteps.train_step(model=model,
                                                train_loader=train_loader,
                                                optimizer=optimizer,
                                                scheduler=scheduler)
        accuracy, classify_report = trainEvalSteps.eval_step(model=model,
                                                             test_loader=test_loader,
                                                             test_data=test_data,
                                                             label_encoder=label_encoder)

        saveModelAndResults.save_model(database_file=database_file,
                                       model_id=model_id,
                                       model=model,
                                       tokenizer=tokenizer,
                                       label_encoder=label_encoder)

        saveToDB.save_model_results_to_database(database_file=database_file,
                                                model_id=model_id,
                                                loss=loss,
                                                accuracy=accuracy,
                                                classify_report=classify_report)

        return 200, "Training Completed"
    else:
        return 400, "Please choose one of the model algorithms available: bert"




def make_single_prediction(database_file, model_id, input_text):
    # Load pre-trained model, tokenizer, and label encoder
    model, tokenizer, label_encoder = processData.load_model_tokenizer_encoder_from_saved_path(
        database_file=database_file,
        model_id=model_id
    )

    # Tokenize and preprocess the input text
    input_tokens = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

    # Forward pass to get predictions
    with torch.no_grad():
        logits = model(**input_tokens).logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class index
    predicted_class_index = torch.argmax(probabilities, dim=1).item()

    # Decode the predicted class
    predicted_class = label_encoder.classes_[predicted_class_index]

    return predicted_class
