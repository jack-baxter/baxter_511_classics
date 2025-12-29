# Classical Music Composer Classification

deep learning models (CNN & LSTM) to classify classical music composers from midi files. currently supports bach, beethoven, chopin, and mozart.

built for AAI-511 final project, converted from colab notebook to local python for better reproducibility and deployment.

## what it does

takes midi files, converts them to piano rolls (basically 2d matrices of pitch over time), then trains neural networks to predict which composer wrote each piece. ended up with two model types:

- **cnn**: treats piano rolls like images, good at catching local patterns and melodic features
- **lstm**: focuses on temporal/sequential patterns, better for long-term musical structure

## dataset

using the `midi_classic_music` dataset from kaggle (originally 3929 files from 175 composers, filtered down to just the big 4 for scope reasons).

each midi file gets normalized to 40 seconds (4000 timesteps at 100hz sample rate) to keep memory usage reasonable and training time manageable.

## project structure

```
composer_classifier/
├── data/                    # midi files organized by composer
│   └── midi_files/
│       ├── Bach/
│       ├── Beethoven/
│       ├── Chopin/
│       └── Mozart/
├── models/                  # saved model files
├── checkpoints/             # training checkpoints
├── logs/                    # training logs
├── data_processing.py       # midi to piano roll conversion
├── models.py               # cnn and lstm architectures
├── train.py                # training utilities
├── evaluate.py             # evaluation metrics
├── config.py               # configuration loader
├── main.py                 # baseline model training
├── main_improved.py        # improved model training (v2)
└── requirements.txt
```

## setup

1. clone the repo
```bash
git clone <your-repo-url>
cd composer_classifier
```

2. install dependencies
```bash
pip install -r requirements.txt
```

3. download midi dataset from [kaggle](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music) and organize into the structure above

4. copy `.env.example` to `.env` and adjust paths if needed
```bash
cp .env.example .env
```

## usage

train baseline models:
```bash
python main.py
```

train improved models with better hyperparameters:
```bash
python main_improved.py
```

models get saved to `./models/` and best checkpoints go to `./checkpoints/`.

## model details

### baseline cnn
- 2 conv blocks (32 and 64 filters)
- maxpooling for dimensionality reduction
- dense layer with dropout (0.5)
- ~10 epochs

### improved cnn (v2)
- 3 conv blocks with batch normalization
- average pooling instead of max (better convergence)
- deeper feature extraction with 128 filters in final conv layer
- ~20 epochs with early stopping

### baseline lstm
- 2 lstm layers (128 -> 64 units)
- focuses on sequential/temporal patterns
- ~10 epochs

### improved lstm (v2)
- 2 lstm layers (128 -> 128 units) with recurrent dropout (0.3)
- increased capacity in second layer to capture more complex patterns
- ~20 epochs with early stopping

## notes and gotchas

- **memory usage**: piano rolls are pretty big (128 x 4000 per file). if you're running into OOM issues, reduce `MAX_*_FILES` in `.env` or decrease `FIXED_LENGTH`
- **training time**: cnn trains faster than lstm on this data. full dataset takes a while even on decent hardware
- **data imbalance**: originally had different sample sizes per composer (20 bach, 20 beethoven, 30 chopin, 40 mozart). kept it that way from original implementation but could affect results
- **reproducibility**: set `RANDOM_STATE=42` for consistent splits across runs

## performance

baseline models hit around 60-70% accuracy (actual numbers depend on data split). improved models with batch norm and additional layers push into 70-80% range.

main confusion is between beethoven and mozart (makes sense - similar era, similar instrumentation). bach is usually most distinct.

## future work

things i'd try if i had more time:
- experiment with different epoch counts (might need more for lstm)
- try residual connections (resnet-style) for deeper cnns
- data augmentation (pitch shifts, time stretches)
- attention mechanisms for lstm
- ensemble methods combining cnn + lstm predictions

## acknowledgments

- dataset: [midi_classic_music on kaggle](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)
- `get_piano_roll()` function originally from xai grok (modified for error handling)
- `evaluate_model()` function from xai grok
- project for AAI-511: neural networks and deep learning (dr. mokhtari)

## license

mit or whatever - it's a school project, use it however
