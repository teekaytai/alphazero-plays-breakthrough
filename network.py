class Network:
    # If directory is not None, loads the network saved in the directory.
    # Else, creates a neural network with randomly initialised weights.
    def __init__(self, directory=None):
        pass

    def save(self, directory):
        pass

    # Returns a copy of this network, including its weights
    def copy(self):
        pass

    def train(self, training_data):
        pass

    # Returns the policy vector and expected value of the given state
    def predict(self, state):
        pass
