import numpy as np

EMPTY = 0
P1 = 1
P2 = -1
N = 8
FORWARD_MOVES = N * (N - 1)
DIAG_MOVES = (N - 1) * (N - 1)
TOTAL_MOVES = FORWARD_MOVES + 2 * DIAG_MOVES
START_BOARD = np.zeros((N, N), dtype=int)
START_BOARD[0] = P2
START_BOARD[1] = P2
START_BOARD[N - 2] = P1
START_BOARD[N - 1] = P1

class Breakthrough:
    def __init__(self, turn=P1, p1_pawns=2*N, p2_pawns=2*N, winner=0, board=None):
        self.turn = turn
        self.p1_pawns = p1_pawns
        self.p2_pawns = p2_pawns
        self.winner = winner
        self.board = board if board is not None else np.copy(START_BOARD)

    @staticmethod
    def num_possible_moves():
        return TOTAL_MOVES

    # Returns the state of this game, from the perspective of the current player. Used
    # as input to the neural network
    def get_state(self):
        state = self.board.flatten()
        return state if self.turn == P1 else -np.flip(state)

    # Returns an int that uniquely identifies the state of this game. Useful for hashing
    def get_id(self):
        h = 0
        for x in self.board.reshape(-1):
            h = 3 * h + int(x)
        return h

    # Returns a vector of 1s and 0s denoting which moves are valid and invalid respectively in the
    # current state from the current player's perspective
    def get_valid_moves(self):
        valid_moves = np.empty(TOTAL_MOVES, dtype=int)
        player_board = self.board if self.turn == P1 else np.flip(self.board)
        pawns = player_board[1:] == self.turn
        valid_forward_moves = np.logical_and(pawns, (player_board[:-1] == EMPTY))
        valid_diag_left_moves = np.logical_and(pawns[:, 1:], (player_board[:-1, :-1] != self.turn))
        valid_diag_right_moves = np.logical_and(pawns[:, :-1], (player_board[:-1, 1:] != self.turn))
        valid_moves[:FORWARD_MOVES] = valid_forward_moves.reshape(-1)
        valid_moves[FORWARD_MOVES:-DIAG_MOVES] = valid_diag_left_moves.reshape(-1)
        valid_moves[-DIAG_MOVES:] = valid_diag_right_moves.reshape(-1)
        return valid_moves

    # Applies the given move to this game, mutating it
    def play_move(self, move):
        player_board = self.board if self.turn == P1 else np.flip(self.board)
        if move < FORWARD_MOVES:
            # Forward move
            r, c = divmod(move, N)
            player_board[r, c] = self.turn
            player_board[r + 1, c] = EMPTY
        elif move < FORWARD_MOVES + DIAG_MOVES:
            # Diagonal left move
            r, c = divmod(move - FORWARD_MOVES, N - 1)
            if self.turn == P1:
                p2_pawns -= player_board[r, c] == P2
            else:
                p1_pawns -= player_board[r, c] == P1
            player_board[r, c] = self.turn
            player_board[r + 1, c + 1] = EMPTY
        else:
            # Diagonal right move
            r, c = divmod(move - FORWARD_MOVES - DIAG_MOVES, N - 1)
            if self.turn == P1:
                p2_pawns -= player_board[r, c + 1] == P2
            else:
                p1_pawns -= player_board[r, c + 1] == P1
            player_board[r, c + 1] = self.turn
            player_board[r + 1, c] = EMPTY
        if r == 0 or self.p1_pawns == 0 or self.p2_pawns == 0:
            self.winner = self.turn
        self.turn = -self.turn

    # Returns a new Breakthrough instance that is the result of playing the given move on this game
    def with_move(self, move):
        new_game = Breakthrough(self.turn, self.p1_pawns, self.p2_pawns, self.winner, np.copy(self.board))
        new_game.play_move(move)
        return new_game

    def is_game_over(self):
        return self.winner != 0

    # Returns 1 if player 1 won, or -1 if player 2 won. Breakthrough never ends in a draw
    def get_result(self):
        return self.winner
