# audio_indices
Audio analysis for realtime field installations on limited bandwidth

## Files included in this repository:

1. `analysis.py` Contains the classifier (currently yamnet) and bioacoustic index functions
2. `audio_file.py` Runs the above analysis functions on a .wav file. Run as ```$ python3 audio_file.py wavfile.wav``` Outputs a `wavfile.npz` file (in order to match the output of `audio_stream.py`
3. `audio_stream.py` Begins a realtime audio stream, and calls `analysis.py` each time a new block of audio is buffered (blocksize set in `params.py`). This outputs a timestamp-titled zip file of numpy arrays (.npz) so as to be able to upload results using as little bandwidth as possible. The .npz file consists of two numpy arrays: *embeddings*, an array of 1024 dimensional embeddings every 0.48 seconds of audio (used for running future classifiers based on targeted research questions), and *scores_indices*, consisting of a single 525 dimensional vector for every block of audio. The first 521 entries are the yamnet classification scores (see yamnet_class_names.csv), and the last 4 are the four bioacoustic indices.
4. `params.py` Sets basic parameters for all other programs.
5. `to_csv.py` Converts .npz output to more usable formats. This saves csv files with the top 10 yamnet classes for each block of audio and with the 4 bioacoustic indices for each block of audio, and internally extracts the embeddings as a numpy array. Run as ```$ python3 to_csv.py data.npz``` and produces files named `data_classes.csv` and `data_indices.csv`
6. `yamnet_class_map.csv` Gives the class names for each class index in the yamnet scores.
7. `yamnet.tflite` The TensorFlow lite yamnet model used for the classifier.

## Sources and Additional Info

Bioacoustic indices come from https://github.com/patriceguyot/Acoustic_Indices, which borrows from the R package https://cran.r-project.org/web/packages/soundecology/vignettes/intro.html

Yamnet is documented in several places:
- https://tfhub.dev/google/yamnet/1
- https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html
