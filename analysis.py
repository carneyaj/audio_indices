#!/usr/bin/env python3
'''
Analysis:
-source classifier
-Soundscape Ecology functions:
	-Acoustic Complexity Index
	-Bioacoustic Index
	-Acoustic Evenness Index
	-Acoustic Diversity Index
'''

from tflite_runtime.interpreter import Interpreter
import numpy as np
import io
import csv
from datetime import datetime

from scipy import signal, fftpack

#from display import displayprint, displayoff

import params

#-----------------------------------------------------------
#     Main Analysis Function

def analysis(data,samplerate):

	waveform = data.flatten()
	timestamp = datetime.now().strftime("%H:%M:%S.%f")
	#filename = timestamp + ".wav"
	#write(filename, samplerate, indata)
	seconds = int(len(waveform)/samplerate)
	#print("---", seconds,"second audio frame at",timestamp)
	
	scores , embeddings, _ = classify(waveform)

	# Pass embeddings to additional classifier here
	# morescores = secondclassifier(embeddings)

	spectro, frequencies, j_bin_mult = compute_spectrogram(waveform,samplerate)

	j_bin = 5 #window width in seconds
	ACI, _ = compute_ACI(spectro, j_bin*j_bin_mult)
	#print("Acoustic Complexity Index:", ACI)

	BI = compute_BI(spectro, frequencies)
	#print("Bioacoustic Index:", BI)

	AEI = compute_AEI(spectro)
	#print("Acoustic Evenness Index:", AEI)

	ADI = compute_ADI(spectro)
	#print("Acoustic Diversity Index:", ADI)

	timestamp = np.array(datetime.now().timestamp())
	bioindices = np.array([ACI,BI,AEI,ADI], dtype = params.out_dtype)
	return timestamp, scores.mean(axis=0, dtype = params.out_dtype), embeddings, bioindices

#-----------------------------------------------------------
#     Classifier Functions

def class_names_from_csv(class_map_csv_text):
	"""Returns list of class names corresponding to score vector."""
	class_map_csv = io.StringIO(class_map_csv_text)
	class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
	class_names = class_names[1:]  # Skip CSV header
	return class_names

interpreter = Interpreter("/home/pi/audio_indices/yamnet.tflite")

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

class_names = class_names_from_csv(open("/home/pi/audio_indices/yamnet_class_map.csv").read())

def classify(waveform):
	#time1 = time.time()
	interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
	interpreter.allocate_tensors()
	interpreter.set_tensor(waveform_input_index, waveform)
	interpreter.invoke()
	scores, embeddings, spectrogram = (
	    interpreter.get_tensor(scores_output_index),
	    interpreter.get_tensor(embeddings_output_index),
	    interpreter.get_tensor(spectrogram_output_index))

	top_classes = np.argsort(scores.mean(axis=0))[::-1][:params.top_classes]
	classlist = ""
	for i in top_classes:
		classlist += class_names[i] + "#"
	classlist = classlist[:-1]
	print(classlist.replace("#", ", "))
#	displayprint(classlist)

	#time2 = time.time()
	#classification_time = np.round(time2-time1, 3)
	#print("Classificaiton Time =", classification_time, "seconds.")

	return scores, embeddings, spectrogram


#-----------------------------------------------------------
#     Acoustic Ecology Functions

def compute_spectrogram(waveform, samplerate, windowLength=512, windowHop= 256, scale_audio=True, square=True, windowType='hanning', centered=False, normalized = False ):
    """
    Compute a spectrogram of an audio signal.
    Return a list of list of values as the spectrogram, and a list of frequencies.
    Keyword arguments:
    file -- the real part (default 0.0)
    Parameters:
    file: an instance of the AudioFile class.
    windowLength: length of the fft window (in samples)
    windowHop: hop size of the fft window (in samples)
    scale_audio: if set as True, the signal samples are scale between -1 and 1 (as the audio convention). If false the signal samples remains Integers (as output from scipy.io.wavfile)
    square: if set as True, the spectrogram is computed as the square of the magnitude of the fft. If not, it is the magnitude of the fft.
    hamming: if set as True, the spectrogram use a correlation with a hamming window.
    centered: if set as true, each resulting fft is centered on the corresponding sliding window
    normalized: if set as true, divide all values by the maximum value
    """

    sig = waveform

    W = signal.get_window(windowType, windowLength, fftbins=False)
    halfWindowLength = int(windowLength/2)

    if centered:
        time_shift = int(windowLength/2)
        times = range(time_shift, len(sig)+1-time_shift, windowHop) # centered
        frames = [sig[i-time_shift:i+time_shift]*W for i in times] # centered frames
    else:
        times = range(0, len(sig)-windowLength+1, windowHop)
        frames = [sig[i:i+windowLength]*W for i in times]

    if square:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:halfWindowLength]**2 for frame in frames]
    else:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:halfWindowLength] for frame in frames]

    spectro=np.transpose(spectro) # set the spectro in a friendly way

    if normalized:
        spectro = spectro/np.max(spectro) # set the maximum value to 1 y

    niquist = samplerate / 2

    frequencies = [e * niquist / float(windowLength / 2) for e in range(halfWindowLength)] # vector of frequency<-bin in the spectrogram
    
    j_bin_mult = int(samplerate/windowHop)

    return spectro, frequencies, j_bin_mult



def compute_ACI(spectro,j_bin):
    """
    Compute the Acoustic Complexity Index from the spectrogram of an audio signal.
    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an avian community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.
    Ported from the soundecology R package.
    spectro: the spectrogram of the audio signal
    j_bin: temporal size of the frame (in samples)
    """

    #times = range(0, spectro.shape[1], j_bin) # relevant time indices
    times = range(0, spectro.shape[1]-10, j_bin) # alternative time indices to follow the R code

    jspecs = [np.array(spectro[:,i:i+j_bin]) for i in times]  # sub-spectros of temporal size j

    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs] 	# list of ACI values on each jspecs
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values # return main (global) value, temporal values

def compute_BI(spectro, frequencies, min_freq = 2000, max_freq = 8000):
    """
    Compute the Bioacoustic Index from the spectrogram of an audio signal.
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in dB) minus the minimum frequency value of this mean spectre.
    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii: bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144.
    spectro: the spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    min_freq: minimum frequency (in Hertz)
    max_freq: maximum frequency (in Hertz)
    Ported from the soundecology R package.
    """

    min_freq_bin = int(np.argmin([abs(e - min_freq) for e in frequencies])) # min freq in samples (or bin)
    max_freq_bin = int(np.ceil(np.argmin([abs(e - max_freq) for e in frequencies]))) # max freq in samples (or bin)

    min_freq_bin = min_freq_bin - 1 # alternative value to follow the R code



    spectro_BI = 20 * np.log10(spectro/np.max(spectro))  #  Use of decibel values. Equivalent in the R code to: spec_left <- spectro(left, f = samplingrate, wl = fft_w, plot = FALSE, dB = "max0")$amp
    spectre_BI_mean = 10 * np.log10 (np.mean(10 ** (spectro_BI/10), axis=1))     # Compute the mean for each frequency (the output is a spectre). This is not exactly the mean, but it is equivalent to the R code to: return(a*log10(mean(10^(x/a))))
    spectre_BI_mean_segment =  spectre_BI_mean[min_freq_bin:max_freq_bin]   # Segment between min_freq and max_freq
    spectre_BI_mean_segment_normalized = spectre_BI_mean_segment - min(spectre_BI_mean_segment) # Normalization: set the minimum value of the frequencies to zero.
    area = np.sum(spectre_BI_mean_segment_normalized / (frequencies[1]-frequencies[0]))   # Compute the area under the spectre curve. Equivalent in the R code to: left_area <- sum(specA_left_segment_normalized * rows_width)

    return area

def gini(values):
    """
    Compute the Gini index of values.
    values: a list of values
    Inspired by http://mathworld.wolfram.com/GiniCoefficient.html and http://en.wikipedia.org/wiki/Gini_coefficient
    """
    y = sorted(values)
    n = len(y)
    G = np.sum([i*j for i,j in zip(y,range(1,n+1))])
    G = 2 * G / np.sum(y) - (n+1)
    return G/n

def compute_AEI(spectro, freq_band_Hz=32, max_freq=8000, db_threshold=-50, freq_step=800):
    """
    Compute Acoustic Evenness Index of an audio signal.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute AEI (in Hertz)
    Ported from the soundecology R package.
    """

    bands_Hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_Hz for f in bands_Hz]

    spec_AEI = 20*np.log10(spectro/np.max(spectro))
    spec_AEI_bands = [spec_AEI[int(bands_bin[k]):int(bands_bin[k]+bands_bin[1]),] for k in range(len(bands_bin))]

    values = [np.sum(spec_AEI_bands[k]>db_threshold)/float(spec_AEI_bands[k].size) for k in range(len(bands_bin))]

    return gini(values)

def compute_ADI(spectro, freq_band_Hz=32,  max_freq=8000, db_threshold=-50, freq_step=800):
    """
    Compute Acoustic Diversity Index.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute ADI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute ADI (in Hertz)
    Ported from the soundecology R package.
    """


    bands_Hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_Hz for f in bands_Hz]

    spec_ADI = 20*np.log10(spectro/np.max(spectro))
    spec_ADI_bands = [spec_ADI[int(bands_bin[k]):int(bands_bin[k]+bands_bin[1]),] for k in range(len(bands_bin))]

    values = [np.sum(spec_ADI_bands[k]>db_threshold)/float(spec_ADI_bands[k].size) for k in range(len(bands_bin))]

    # Shannon Entropy of the values
    #shannon = - sum([y * np.log(y) for y in values]) / len(values)  # Follows the R code. But log is generally log2 for Shannon entropy. Equivalent to shannon = False in soundecology.

    # The following is equivalent to shannon = True (default) in soundecology. Compute the Shannon diversity index from the R function diversity {vegan}.
    #v = [x/np.sum(values) for x in values]
    #v2 = [-i * j  for i,j in zip(v, np.log(v))]
    #return np.sum(v2)

    # Remove zero values (Jan 2016)
    values = [value for value in values if value != 0]

    #replace zero values by 1e-07 (closer to R code, but results quite similars)
    #values = [x if x != 0 else 1e-07 for x in values]

    return np.sum([-i/ np.sum(values) * np.log(i / np.sum(values))  for i in values])




