import torch
import torch.nn as nn
from hankelization import prepare_hankel_data

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'dense_lstm_fusion_model.pth'

# data
def prepare_data():
    return prepare_hankel_data()

# temporal attention - learns which timesteps in the LSTM output matter most
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_size)
        scores = self.attn(lstm_out)           # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1) # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1) # (batch, hidden_size)
        return context, weights.squeeze(-1)

# model
class DenseLSTMFusion(nn.Module):
    def __init__(self, micro_dim, macro_dim):
        super().__init__()

        # micro branch - standard vanilla LSTM
        self.micro_lstm = nn.LSTM(
            input_size=micro_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # macro branch - 2 layer stacked vanilla LSTM
        self.macro_lstm = nn.LSTM(
            input_size=macro_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # temporal attention for each branch
        self.micro_attn = TemporalAttention(64)
        self.macro_attn = TemporalAttention(128)

        self.micro_fc = nn.Linear(64, 1)
        self.macro_fc = nn.Linear(128, 1)

        # cross-branch attention gate - learns micro vs macro importance
        self.gate = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # fusion branch
        self.fusion_fc = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, xm, xM):
        # run LSTMs and get full output sequences
        micro_seq, _ = self.micro_lstm(xm)   # (batch, seq, 64)
        macro_seq, _ = self.macro_lstm(xM)   # (batch, seq, 128)

        # apply temporal attention instead of taking last hidden state
        h1, _ = self.micro_attn(micro_seq)   # (batch, 64)
        h2, _ = self.macro_attn(macro_seq)   # (batch, 128)

        micro_out, macro_out = self.micro_fc(h1), self.macro_fc(h2)

        # cross-branch gated fusion
        combined = torch.cat([h1, h2], dim=1)   # (batch, 192)
        gate_weights = self.gate(combined)       # (batch, 2)

        # scale each branch by its learned gate weight
        h1_gated = h1 * gate_weights[:, 0:1]    # (batch, 64)
        h2_gated = h2 * gate_weights[:, 1:2]    # (batch, 128)

        fusion = torch.cat([h1_gated, h2_gated], dim=1)  # (batch, 192)
        final_out = self.fusion_fc(fusion)

        return micro_out, macro_out, final_out