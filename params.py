#!/usr/bin/env python3
'''

Call analysis functions on an existing .wav file
Resample to 16kHz

'''

samplerate = 16000	# Audio samplerate
seconds = 2		# length of each window of analysis
blocks = 300		# number of analysis blocks of length *seconds* 
gain = 20   		# multiplier on audio samples

save_directory = "/home/pi/data/"
out_dtype = "float32"

top_classes = 4

