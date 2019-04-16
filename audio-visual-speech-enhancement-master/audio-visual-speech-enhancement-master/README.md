# Visual Speech Enhancement
Implementation of the method described in the paper: [Visual Speech Enhancement](http://www.vision.huji.ac.il/visual-speech-enhancement) by Aviv Gabbay, Asaph Shamir and Shmuel Peleg.

## Speech Enhancement Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=nyYarDGpcYA" target="_blank">
<img src="http://img.youtube.com/vi/nyYarDGpcYA/0.jpg" width="480" height="360" />
</a>

## Usage
### Dependencies
* python >= 2.7
* [mediaio](https://github.com/avivga/mediaio)
* [face-detection](https://github.com/avivga/face-detection)
* keras >= 2.0.4
* numpy >= 1.12.1
* dlib >= 19.4.0
* opencv >= 3.2.0
* librosa >= 0.5.1

### Getting started
Given an audio-visual dataset of the directory structure:
```
├── speaker-1
|   ├── audio
|   |   ├── f1.wav
|   |   └── f2.wav
|   └── video
|	├── f1.mp4
|	└── f2.mp4
├── speaker-2
|   ├── audio
|   |   ├── f1.wav
|   |   └── f2.wav
|   └── video
|	├── f1.mp4
|	└── f2.mp4
...
```
and noise directory contains audio files (*.wav) of noise samples, do the following steps.

Preprocess train, validation and test datasets separately by:
```
speech_enhancer.py --base_dir <output-dir-path> preprocess
    --data_name <preprocessed-data-name>
    --dataset_dir <dataset-dir-path>
    --noise_dirs <noise-dir-path> ...
    [--speakers <speaker-id> ...]
    [--ignored_speakers <speaker-id> ...] 
```

Then, train the model by:
```
speech_enhancer.py --base_dir <output-dir-path> train
    --model <model-name>
    --train_data_names <preprocessed-training-data-name> ...
    --validation_data_names <preprocessed-validation-data-name> ...
    [--gpus <num-of-gpus>]
```

Finally, enhance the test noisy speech samples by:
```
speech_enhancer.py --base_dir <output-dir-path> predict
    --model <model-name>
    --data_name <preprocessed-test-data-name>
    [--gpus <num-of-gpus>]
```

## Citing
If you find this project useful for your research, please cite
```
@inproceedings{gabbay2018visual,
  author    = {Aviv Gabbay and
               Asaph Shamir and
               Shmuel Peleg},
  title     = {Visual Speech Enhancement},
  booktitle = {Interspeech},
  pages     = {1170--1174},
  publisher = {{ISCA}},
  year      = {2018}
}
```
