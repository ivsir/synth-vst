import numpy as np

# Flatten the chord data into a list of tuples (midiKeys, chordName)
chords = []
for root, variations in chords_data.items():
    for variation, details in variations.items():
        chords.append((details['midiKeys'], details['name']))

# Prepare input and output data
# Convert midiKeys to a binary vector (88-key piano)
def midi_to_vector(midi_keys):
    vector = np.zeros(88)
    for key in midi_keys:
        vector[key - 21] = 1
    return vector

X = np.array([midi_to_vector(midiKeys) for midiKeys, _ in chords])
y = np.array([name for _, name in chords])

# Encode chord names to numerical labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
