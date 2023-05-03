import torch
import torch.nn as nn

class DurationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(DurationPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, phoneme_embeddings, phoneme_lengths):
        phoneme_lengths, sort_idx = phoneme_lengths.sort(0, descending=True)
        phoneme_embeddings = phoneme_embeddings[sort_idx]

        packed_input = nn.utils.rnn.pack_padded_sequence(phoneme_embeddings,
                                                         phoneme_lengths.cpu().numpy(),
                                                         enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = self.linear(output)
        output = output.squeeze(2)[sort_idx]
        return output

class PitchPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(PitchPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, phoneme_embeddings, phoneme_lengths):
        phoneme_lengths, sort_idx = phoneme_lengths.sort(0, descending=True)
        phoneme_embeddings = phoneme_embeddings[sort_idx]

        packed_input = nn.utils.rnn.pack_padded_sequence(phoneme_embeddings,
                                                         phoneme_lengths.cpu().numpy(),
                                                         enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = self.linear(output)
        output = output.squeeze(2)[sort_idx]
        return output

# Train both models with L1 loss
criterion = nn.L1Loss()
duration_predictor = DurationPredictor(input_size, hidden_size, num_layers)
pitch_predictor = PitchPredictor(input_size, hidden_size, num_layers)
duration_predictor_optimizer = torch.optim.Adam(duration_predictor.parameters())
pitch_predictor_optimizer = torch.optim.Adam(pitch_predictor.parameters())

for epoch in range(num_epochs):
    for phoneme_embeddings, durations, pitch, phoneme_lengths in train_loader:
        duration_predictions = duration_predictor(phoneme_embeddings, phoneme_lengths)
        pitch_predictions = pitch_predictor(phoneme_embeddings, phoneme_lengths)

        duration_loss = criterion(duration_predictions, durations)
        pitch_loss = criterion(pitch_predictions, pitch)

        duration_predictor_optimizer.zero_grad()
        duration_loss.backward()
        duration_predictor_optimizer.step()

        pitch_predictor_optimizer.zero_grad()
        pitch_loss.backward()
        pitch_predictor_optimizer.step()
