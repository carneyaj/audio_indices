#!/usr/bin/env python3
'''

Call analysis functions on an existing .wav file
Resample to 16kHz


'''
import sys
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read, write
import scipy.signal as sps
import params

from analysis import *


#duration = 60  # seconds
#myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
#sd.wait()
#write("test.wav",16000,myrecording)


filename = sys.argv[1]

emb = np.empty([0,1024],dtype = params.out_dtype) #Should initialize to fullsize for runtime eventually, fill with zeros
classes_and_indices = np.empty([0,525],dtype = params.out_dtype)

outfilename = filename[:-4]

try:
	old_sr, data = read(filename)
	if data.dtype == "int16":
		data = data.astype("float32")/(2 **15)
		print("int16")
	elif data.dtype == "int32":
		data = data/(2 ** 31)
		print("int32")
	else:
		print(data.dtype)

	print("resampling to 16kHz...")
	if old_sr != 16000:
		number_of_samples = round(len(data) * float(params.samplerate) / old_sr)
		data = sps.resample(data, number_of_samples)

	data = data.flatten()
	print("resampling done")

	for i in range(0,len(data),params.samplerate * params.seconds):
		waveform = data[i:i+params.samplerate * params.seconds]
		if len(waveform) == params.samplerate * params.seconds:
			print(np.max(data))
			timestamp, scores, embeddings, bioindices = analysis(waveform,params.samplerate)
			#newdata = np.append(timestamp, scores)
			newdata = np.append(scores,bioindices)
			classes_and_indices = np.vstack((classes_and_indices, newdata))
			emb = np.vstack((emb,embeddings.astype(params.out_dtype)))


	print("\nWriting Data...")
	np.savez_compressed(params.save_directory + outfilename, embeddings=emb, scores_indices=classes_and_indices)
	print("Done")
except KeyboardInterrupt:
	displayoff()
	print("\nquitting...")
except Exception as e:
	print(type(e).__name__ + ': ' + str(e))
	displayoff()
	print("\nquitting...")
