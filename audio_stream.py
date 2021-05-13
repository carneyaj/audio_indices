#!/usr/bin/env python3
'''

Create an audio stream then call analysis functions on it


'''

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
from queue import Queue
from analysis import *
import params


#-----------------------------------------------------------
#     Recording Stream

queue = Queue(maxsize=1)

emb = np.empty([0,1024],dtype = params.out_dtype) #Should initialize to fullsize for runtime eventually, fill with zeros
classes_and_indices = np.empty([0,525],dtype = params.out_dtype)


filename = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

count = 0

try:

	def callback(indata, frames, time, status):
		queue.put(indata.copy())

	with sd.InputStream(channels=1, callback=callback, blocksize=int(params.seconds * params.samplerate), samplerate=params.samplerate):
		while count < params.blocks:
			timestamp, scores, embeddings, bioindices = analysis(params.gain*queue.get(),params.samplerate)
			#newdata = np.append(timestamp, scores)
			newdata = np.append(scores,bioindices)
			classes_and_indices = np.vstack((classes_and_indices, newdata))
			emb = np.vstack((emb,embeddings.astype(params.out_dtype)))
			count += 1
			print(count*params.seconds/60, "minutes of audio")
			
		print("\nwriting datafile...")
		np.savez_compressed(params.save_directory + filename, embeddings=emb, scores_indices=classes_and_indices)
		print("done")


#-----------------------------------------------------------
#   Error handling

except KeyboardInterrupt:
#	displayoff()
	print("\nwriting datafile...")
	np.savez_compressed(params.save_directory + filename, embeddings=emb, scores_indices=classes_and_indices)
	print("done")
except Exception as e:
	print(type(e).__name__ + ': ' + str(e))
#	displayoff()
	print("\nquitting...")
