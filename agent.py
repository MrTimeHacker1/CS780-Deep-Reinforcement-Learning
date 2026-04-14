"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (a2c_lstm_obelix_wandb.pth or weights.pth)
inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation while maintaining the LSTM hidden state.

The evaluator will import this file and call `policy(obs, rng)`.
"""


# THIS TIME I AM USING A2C + LSTM + IMITATION + REWARD SHAPING

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # stores the loaded model
_HX = None     # stores the LSTM hidden state


class ActorCriticLSTM(nn.Module):
    """Network architecture matching the training script."""
    def __init__(self, obs_dim=18, num_actions=5, mlp_size=64, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, mlp_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=mlp_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs, hx):
        x = self.encoder(obs)
        x = x.unsqueeze(1)
        lstm_out, hx_new = self.lstm(x, hx)
        lstm_out = lstm_out.squeeze(1)

        logits = self.actor(lstm_out)
        value = self.critic(lstm_out).view(-1)
        return logits, value, hx_new

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        return (h, c)


def _load_once():
    """Load the trained model and initialize the hidden state."""
    global _MODEL, _HX
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)

    # Check for the specific wandb filename first, fallback to weights.pth
    wpath = os.path.join(submission_dir, "weights.pth")

    model = ActorCriticLSTM(obs_dim=18, num_actions=5, mlp_size=64, hidden_size=256)

    # Load weights (map_location handles CPU evaluation if trained on GPU)
    model.load_state_dict(torch.load(wpath, map_location=torch.device('cpu')))
    model.eval()

    _MODEL = model

    # Initialize the LSTM hidden state for a single environment (batch_size=1)
    _HX = model.init_hidden(1)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained recurrent model to choose the best action."""
    global _HX
    _load_once()
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits, _, _HX = _MODEL(x, _HX)
        logits = logits.squeeze(0).numpy()

    # Select the action with the highest probability
    return ACTIONS[int(np.argmax(logits))]