#!/usr/bin/env python3

import numpy as np
from config import load_config, print_config
from data_processing import load_all_data, prepare_training_data
from models import build_cnn_model, build_lstm_model, compile_model
from train import split_data, prepare_cnn_data, train_model, save_model
from evaluate import evaluate_model, compare_models

def main():
    #load configuration from env file
    config = load_config()
    print_config(config)
    
    #load and preprocess midi files into piano rolls
    print("\nloading data...")
    df = load_all_data(config)
    
    if df.empty:
        print("error: no data loaded, check data directories")
        return
    
    #convert to numpy arrays and one-hot encode labels
    x, y = prepare_training_data(df)
    
    #split train/test with reproducible random state
    xtrain, xtest, ytrain, ytest = split_data(
        x, y, 
        test_size=config['test_split'],
        random_state=config['random_state']
    )
    
    #prepare data for cnn (add channel dimension)
    xtraincnn = prepare_cnn_data(xtrain)
    xtestcnn = prepare_cnn_data(xtest)
    
    results = {}
    
    #train baseline cnn model
    print("\ntraining cnn model...")
    cnn_model = build_cnn_model((xtrain.shape[1], xtrain.shape[2], 1))
    cnn_model = compile_model(cnn_model)
    cnn_model, _ = train_model(
        cnn_model, xtraincnn, ytrain,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        checkpoint_path=f"{config['checkpoint_dir']}/cnn_best.keras"
    )
    
    #evaluate cnn
    cnn_metrics = evaluate_model(cnn_model, xtestcnn, ytest, "cnn baseline")
    results['cnn_baseline'] = cnn_metrics
    
    #save cnn model
    save_model(cnn_model, f"{config['model_save_dir']}/cnn_model.keras")
    
    #train baseline lstm model
    print("\ntraining lstm model...")
    lstm_model = build_lstm_model((xtrain.shape[1], xtrain.shape[2]))
    lstm_model = compile_model(lstm_model)
    lstm_model, _ = train_model(
        lstm_model, xtrain, ytrain,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        checkpoint_path=f"{config['checkpoint_dir']}/lstm_best.keras"
    )
    
    #evaluate lstm
    lstm_metrics = evaluate_model(lstm_model, xtest, ytest, "lstm baseline")
    results['lstm_baseline'] = lstm_metrics
    
    #save lstm model
    save_model(lstm_model, f"{config['model_save_dir']}/lstm_model.keras")
    
    #compare model performance
    compare_models(results)

if __name__ == "__main__":
    main()
