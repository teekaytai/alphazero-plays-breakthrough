class Breakthrough:
    def __init__(self):
        pass

    @staticmethod
    def num_possible_moves():
        pass

    @staticmethod
    def all_possible_moves():
        pass

    # Returns the state of this game, from the perspective of the current player. Used
    # as input to the neural network
    def get_state(self):
        pass

    # Returns an int that uniquely identifies the state of this game. Useful for hashing
    def get_id(self):
        pass

    # Returns a vector of 1s and 0s denoting which moves are valid and invalid respectively in the
    # current state from the current player's perspective
    def get_valid_moves(self):
        pass

    # Applies the given move to this game, mutating it
    def play_move(self, move):
        pass

    # Returns a new Breakthrough instance that is the result of playing the given move on this game
    def with_move(self, move):
        pass

    def is_game_over(self):
        pass

    # Returns 1 if player 1 won, or -1 if player 2 won. Breakthrough never ends in a draw
    def get_result(self):
        pass
