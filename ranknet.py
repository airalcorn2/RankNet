# Michael A. Alcorn (malcorn@redhat.com)
# A (slightly modified) implementation of RankNet as described in [1].
#   [1] http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
#   [2] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

import numpy as np

from keras import backend
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model

INPUT_DIM = 50

# Model.
h_1 = Dense(128, activation = "relu")
h_2 = Dense(64, activation = "relu")
h_3 = Dense(32, activation = "relu")
s = Dense(1)

# Relevant document score.
rel_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_rel = h_1(rel_doc)
h_2_rel = h_2(h_1_rel)
h_3_rel = h_3(h_2_rel)
rel_score = s(h_3_rel)

# Irrelevant document score.
irr_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_irr = h_1(irr_doc)
h_2_irr = h_2(h_1_irr)
h_3_irr = h_3(h_2_irr)
irr_score = s(h_3_irr)

# Subtract scores.
diff = Subtract()([rel_score, irr_score])

# Pass difference through sigmoid function.
prob = Activation("sigmoid")(diff)

# Build model.
model = Model(inputs = [rel_doc, irr_doc], outputs = prob)
model.compile(optimizer = "adadelta", loss = "binary_crossentropy")

# Fake data.
N = 100
X_1 = 2 * np.random.uniform(size = (N, INPUT_DIM))
X_2 = np.random.uniform(size = (N, INPUT_DIM))
y = np.ones((X_1.shape[0], 1))

# Train model.
NUM_EPOCHS = 10
BATCH_SIZE = 10
history = model.fit([X_1, X_2], y, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, verbose = 1)

# Generate scores from document/query features.
get_score = backend.function([rel_doc], [rel_score])
get_score([X_1])
get_score([X_2])
