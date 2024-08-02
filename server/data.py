import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from index import load_chords

# Load the chord data
chords_data = load_chords()

if chords_data is not None:
    # Flatten the chord data into a list of tuples (midiKeys, chordName)
    chords = []
    for root, variations in chords_data.items():
        for variation, details in variations.items():
            # chords.append((details['midiKeys'], details['name']))
            chords.append((details['notes'], details['name']))

    # Prepare input and output data
    # Convert midiKeys to a binary vector (88-key piano)
    def midi_to_vector(midi_keys):
        vector = np.zeros(88)
        for key in midi_keys:
            vector[key - 21] = 1
        return vector
  
    X = np.array([midi_to_vector(midiKeys) for midiKeys, _ in chords])
    y = np.array([name for _, name in chords])

    from sklearn.preprocessing import LabelEncoder

    # Encode chord names to numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save the label_encoder to a file
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Now X and y_encoded are ready for further processing, e.g., machine learning
    print("Data prepared successfully.")
else:
    print("Failed to load chord data.")

# Load the label_encoder from the file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

class ChordClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ChordClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_size = 88
num_classes = len(label_encoder.classes_)

model = ChordClassifier(input_size, num_classes)


# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
def notes_to_midi_keys(notes):
    note_to_midi = {
        'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65, 'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71,
        'Db': 61, 'Eb': 63, 'Gb': 66, 'Ab': 68, 'Bb': 70
    }
    midi_keys = []
    for note in notes:
        note_name = note[:-1]  # Remove the octave number
        octave = int(note[-1])  # Get the octave number
        midi_key = note_to_midi.get(note_name)
        if midi_key is not None:
            midi_keys.append(midi_key + (octave - 4) * 12)
    return midi_keys

# def predict_chord(midi_keys):
#     vector = torch.tensor(midi_to_vector(midi_keys), dtype=torch.float32).unsqueeze(0)
#     output = model(vector)
#     _, predicted = torch.max(output, 1)
#     chord_name = label_encoder.inverse_transform(predicted.numpy())
#     return chord_name[0]
def predict_chord(notes):
    midi_keys = notes_to_midi_keys(notes)
    vector = torch.tensor(midi_to_vector(midi_keys), dtype=torch.float32).unsqueeze(0)
    output = model(vector)
    _, predicted = torch.max(output, 1)
    chord_name = label_encoder.inverse_transform(predicted.numpy())
    return chord_name[0]


# Example usage
# example_notes = ["C", "E", "G"]
# predicted_chord = predict_chord(example_notes)
# print(f'Predicted chord: {predicted_chord}')