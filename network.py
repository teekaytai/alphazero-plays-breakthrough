import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import trange

from breakthrough_net import BreakThroughNet as bnnet

logger = logging.getLogger()

# A wrapper class for the breakthrough neural network.
# It has methods to save, load, train, predict and copy the network.
class Network():
    # If directory is not None, loads the network saved in the directory.
    # Else, creates a neural network with randomly initialised weights.
    def __init__(self, path=None):
        self.nnet = bnnet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)

        if path is not None and os.path.exists(path):
            checkpoint = torch.load(path)
            self.nnet.load_state_dict(checkpoint['model_state_dict'])
            logger.debug(f"Model loaded from {path}")

    def save(self, path="checkpoint.pth", optimizer=None, iteration=0):
        """Save the network and optimizer state to the specified path."""
        save_dict = {
            'iteration': iteration,
            'model_state_dict': self.nnet.state_dict(),
        }
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(save_dict, path)
        logger.debug(f"Model saved to {path}")

    def train(self, training_data, batch_size=64, epochs=10, lr=0.001):
        """Train the network using examples from self-play.

        Args:
            training_data: List of (state, policy_target, value_target) tuples
            batch_size: Mini-batch size for training
            epochs: Number of epochs to train
            weight_decay: L2 regularization strength

        Returns:
            Average loss over training
        """
        self.nnet.train()  # Set model to training mode

        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=lr)
        # Extract training data
        states, policy_targets, value_targets = zip(*training_data)

        states = torch.FloatTensor(np.stack(states, axis=0)).to(self.device)
        policy_targets = torch.FloatTensor(np.stack(policy_targets, axis=0)).to(self.device)
        value_targets = torch.FloatTensor(np.array(value_targets)).view(-1, 1).to(self.device)

        # Calculate how many mini-batches we'll have
        n_samples = len(states)
        indices = torch.randperm(n_samples)

        total_loss = 0

        for epoch in trange(epochs):
            epoch_loss = 0

            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                # Get batch indices
                batch_indices = indices[i:i+batch_size]

                # Prepare batch data
                state_batch = states[batch_indices]
                policy_batch = policy_targets[batch_indices]
                value_batch = value_targets[batch_indices]

                # Forward pass
                policy_logits, value = self.nnet(state_batch)

                # Compute loss
                policy_loss = -(policy_batch * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(value, value_batch)
                loss = policy_loss + value_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track losses
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / (n_samples // batch_size)
            logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss

        return total_loss / epochs

    def predict(self, state):
        """Perform inference on a state.

        Args:
            state: A game state (should be properly formatted for the network)

        Returns:
            policy: Probability distribution over actions
            value: Expected value of the state
        """
        self.nnet.eval()  # Set model to evaluation mode

        state = torch.FloatTensor(state).to(self.device)

        # Add batch dimension if not present
        if len(state.shape) == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.nnet(state)

            # Convert policy logits to probabilities
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu()

            # Get scalar value
            value = value.item()

        return policy.detach().numpy(), value

    def copy(self):
        """Create a deep copy of this network with identical weights."""
        new_model = Network()
        new_model.nnet.load_state_dict(self.nnet.state_dict())
        return new_model
