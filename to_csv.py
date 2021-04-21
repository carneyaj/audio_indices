#!/usr/bin/env python3
'''

Convert .npz to a csv for more convenient use

'''

import numpy as np
import params
import sys

filename = sys.argv[1]

filename_prefix = filename[:-4] + "_"

data = np.load(filename)
embeddings = data["embeddings"]

scores = data["scores_indices"][:,:521]
indices = data["scores_indices"][:,521:525]

top_classes = np.argsort(scores, axis = 1)[:,:-params.top_classes - 1:-1].astype('int')

np.savetxt(params.save_directory + filename_prefix + "classes.csv", top_classes, delimiter=",", fmt='%i')
np.savetxt(params.save_directory + filename_prefix + "indices.csv", indices, delimiter=",")