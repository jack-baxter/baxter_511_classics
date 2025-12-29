import os
from dotenv import load_dotenv

#load configuration from environment variables
#returns dict with all hyperparameters and paths for reproducibility
def load_config() -> dict:
    load_dotenv()
    
    config = {
        'bach_dir': os.getenv('BACH_DIR', './data/midi_files/Bach'),
        'beethoven_dir': os.getenv('BEETHOVEN_DIR', './data/midi_files/Beethoven'),
        'chopin_dir': os.getenv('CHOPIN_DIR', './data/midi_files/Chopin'),
        'mozart_dir': os.getenv('MOZART_DIR', './data/midi_files/Mozart'),
        'sample_rate': int(os.getenv('SAMPLE_RATE', 100)),
        'fixed_length': int(os.getenv('FIXED_LENGTH', 4000)),
        'pitch_range': int(os.getenv('PITCH_RANGE', 128)),
        'batch_size': int(os.getenv('BATCH_SIZE', 32)),
        'epochs': int(os.getenv('EPOCHS', 10)),
        'test_split': float(os.getenv('TEST_SPLIT', 0.2)),
        'random_state': int(os.getenv('RANDOM_STATE', 42)),
        'model_save_dir': os.getenv('MODEL_SAVE_DIR', './models'),
        'checkpoint_dir': os.getenv('CHECKPOINT_DIR', './checkpoints'),
        'log_dir': os.getenv('LOG_DIR', './logs'),
        'max_bach_files': int(os.getenv('MAX_BACH_FILES', 20)) if os.getenv('MAX_BACH_FILES') else None,
        'max_beethoven_files': int(os.getenv('MAX_BEETHOVEN_FILES', 20)) if os.getenv('MAX_BEETHOVEN_FILES') else None,
        'max_chopin_files': int(os.getenv('MAX_CHOPIN_FILES', 30)) if os.getenv('MAX_CHOPIN_FILES') else None,
        'max_mozart_files': int(os.getenv('MAX_MOZART_FILES', 40)) if os.getenv('MAX_MOZART_FILES') else None,
    }
    
    return config

#print current configuration for debugging and audit trails
def print_config(config: dict):
    print("current configuration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)
