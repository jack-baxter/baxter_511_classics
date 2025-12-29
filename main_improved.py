#!/usr/bin/env python3

import numpy as np
from config import load_config, print_config
from data_processing import load_all_data, prepare_training_data
from models import build_cnn_model_v2, build_lstm_model_v2, compile_model
from train import split_data, prepare_cnn_data, train_model, save_model
from evaluate import evaluate_model, compare_models, detailed_evaluation

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
    
    #train improved cnn model with batch norm and average pooling
    #increased epochs to 20 for better convergence
    print("\ntraining cnn v2 model...")
    cnn_model_v2 = build_cnn_model_v2((xtrain.shape[1], xtrain.shape[2], 1))
    cnn_model_v2 = compile_model(cnn_model_v2)
    cnn_model_v2, _ = train_model(
        cnn_model_v2, xtraincnn, ytrain,
        epochs=20,
        batch_size=config['batch_size'],
        checkpoint_path=f"{config['checkpoint_dir']}/cnn_v2_best.keras",
        early_stop=True
    )
    
    #evaluate cnn v2
    cnn_v2_metrics = evaluate_model(cnn_model_v2, xtestcnn, ytest, "cnn v2")
    results['cnn_v2'] = cnn_v2_metrics
    
    #detailed evaluation for debugging
    detailed_evaluation(cnn_model_v2, xtestcnn, ytest)
    
    #save cnn v2 model
    save_model(cnn_model_v2, f"{config['model_save_dir']}/cnn_model_v2.keras")
    
    #train improved lstm model with recurrent dropout
    #increased epochs to 20 for better convergence
    print("\ntraining lstm v2 model...")
    lstm_model_v2 = build_lstm_model_v2((xtrain.shape[1], xtrain.shape[2]))
    lstm_model_v2 = compile_model(lstm_model_v2)
    lstm_model_v2, _ = train_model(
        lstm_model_v2, xtrain, ytrain,
        epochs=20,
        batch_size=config['batch_size'],
        checkpoint_path=f"{config['checkpoint_dir']}/lstm_v2_best.keras",
        early_stop=True
    )
    
    #evaluate lstm v2
    lstm_v2_metrics = evaluate_model(lstm_model_v2, xtest, ytest, "lstm v2")
    results['lstm_v2'] = lstm_v2_metrics
    
    #detailed evaluation for debugging
    detailed_evaluation(lstm_model_v2, xtest, ytest)
    
    #save lstm v2 model
    save_model(lstm_model_v2, f"{config['model_save_dir']}/lstm_model_v2.keras")
    
    #compare model performance
    compare_models(results)

if __name__ == "__main__":
    main()
