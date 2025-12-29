import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

#split data into train and test sets with reproducible random state
#default 80/20 split consistent with original implementation
def split_data(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    
    print(f"train set: {xtrain.shape[0]} samples")
    print(f"test set: {xtest.shape[0]} samples")
    
    return xtrain, xtest, ytrain, ytest

#expand dimensions for cnn input format
#adds channel dimension to shape (n, 128, 4000) -> (n, 128, 4000, 1)
def prepare_cnn_data(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, axis=-1)

#train model with validation split and optional checkpointing
#saves best model based on validation loss to prevent overfitting
#includes early stopping to avoid wasting compute on plateaued training
def train_model(model: Sequential, xtrain: np.ndarray, ytrain: np.ndarray,
                epochs: int = 10, batch_size: int = 32, 
                validation_split: float = 0.2,
                checkpoint_path: str = None,
                early_stop: bool = False) -> Sequential:
    
    callbacks = []
    
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    if early_stop:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    history = model.fit(
        xtrain, ytrain,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks if callbacks else None
    )
    
    return model, history

#save trained model to disk for deployment or inference
#uses keras native format for compatibility
def save_model(model: Sequential, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"model saved to {path}")
