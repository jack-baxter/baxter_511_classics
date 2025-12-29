import os
import numpy as np
import pandas as pd
import pretty_midi
from typing import Optional, List, Dict

#convert midi file to normalized piano roll with fixed dimensions
#handles variable length files via zero padding, normalizes to 0-1 range
#original function from xai grok llm, modified for error handling
def get_piano_roll(midi_path: str, fs: int = 100, fixed_length: int = 4000, 
                   pitch_range: int = 128) -> Optional[np.ndarray]:
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        piano_roll = piano_roll / 127.0
        
        current_length = piano_roll.shape[1]
        if current_length < fixed_length:
            padded = np.zeros((pitch_range, fixed_length))
            padded[:, :current_length] = piano_roll
            return padded
        else:
            return piano_roll[:, :fixed_length]
            
    except Exception as e:
        print(f"error processing {midi_path}: {e}")
        return None

#load all midi files from specified directory
#returns list of dicts with piano roll and composer id
#fails gracefully on corrupt files, logs errors for monitoring
def load_composer_data(data_dir: str, composer_id: int, fs: int = 100, 
                       fixed_length: int = 4000, max_files: Optional[int] = None) -> List[Dict]:
    data = []
    
    if not os.path.exists(data_dir):
        print(f"warning: directory not found {data_dir}")
        return data
    
    files = os.listdir(data_dir)
    if max_files:
        files = files[:max_files]
    
    for file_name in files:
        midi_path = os.path.join(data_dir, file_name)
        piano_roll = get_piano_roll(midi_path, fs, fixed_length)
        
        if piano_roll is not None:
            data.append({
                'song': piano_roll,
                'composer_id': composer_id
            })
    
    print(f"loaded {len(data)} files from {data_dir}")
    return data

#aggregate all composer data into single dataframe
#composer ids: 1=bach, 2=beethoven, 3=chopin, 4=mozart
#returns dataframe with shape (n_samples, 2) where song col contains piano rolls
def load_all_data(config: Dict) -> pd.DataFrame:
    all_data = []
    
    composers = [
        (config['bach_dir'], 1, config.get('max_bach_files')),
        (config['beethoven_dir'], 2, config.get('max_beethoven_files')),
        (config['chopin_dir'], 3, config.get('max_chopin_files')),
        (config['mozart_dir'], 4, config.get('max_mozart_files'))
    ]
    
    for data_dir, comp_id, max_files in composers:
        composer_data = load_composer_data(
            data_dir, 
            comp_id,
            config['sample_rate'],
            config['fixed_length'],
            max_files
        )
        all_data.extend(composer_data)
    
    df = pd.DataFrame(all_data)
    print(f"total dataset size: {len(df)} samples")
    
    return df

#prepare training data from dataframe
#converts to numpy arrays and applies one-hot encoding to labels
#returns x with shape (n, 128, 4000) and y with shape (n, 4)
def prepare_training_data(df: pd.DataFrame, num_classes: int = 4) -> tuple:
    from tensorflow.keras.utils import to_categorical
    
    x = np.array(df['song'].tolist())
    y = df['composer_id'].values - 1
    y = to_categorical(y, num_classes=num_classes)
    
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    
    return x, y
