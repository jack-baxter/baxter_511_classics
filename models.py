from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, LSTM, BatchNormalization, 
                                     Activation, AveragePooling2D)

#baseline cnn architecture for piano roll classification
#input expects (batch, 128, 4000, 1) shaped data
#architecture: 2 conv blocks with maxpool, dense layers with dropout
def build_cnn_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    return model

#improved cnn with batch normalization and average pooling
#better convergence properties and reduced overfitting risk
#adds third conv block for deeper feature extraction
def build_cnn_model_v2(input_shape: tuple) -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        AveragePooling2D((2, 2)),
        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        AveragePooling2D((2, 2)),
        Conv2D(128, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        AveragePooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    return model

#baseline lstm for sequential pattern recognition in piano rolls
#input expects (batch, 128, 4000) shaped data without channel dim
#two lstm layers to capture temporal dependencies in music
def build_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    return model

#improved lstm with dropout in recurrent layers
#helps prevent overfitting on temporal patterns
#increased second lstm size to 128 for more complex pattern capture
def build_lstm_model_v2(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True, dropout=0.3),
        LSTM(128, dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    return model

#compile model with standard adam optimizer and categorical crossentropy
#accuracy metric for monitoring during training
def compile_model(model: Sequential) -> Sequential:
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
