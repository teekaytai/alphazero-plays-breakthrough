import numpy as np

EMPTY = False
PAWN = True
P1 = 1
P2 = -1
N = 8
FORWARD_MOVES = N * (N - 1)
DIAG_MOVES = (N - 1) * (N - 1)
TOTAL_MOVES = FORWARD_MOVES + 2 * DIAG_MOVES
START_BOARD = np.zeros((2, N, N), dtype=bool)
START_BOARD[0, -2:] = PAWN
START_BOARD[1, :2] = PAWN

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

    def copy(self):
        return Breakthrough(self.turn, self.p1_pawns, self.p2_pawns, self.winner, np.copy(self.board))

    # Returns the state of this game, from the perspective of the current player. Used
    # as input to the neural network
    def get_state(self):
        return np.copy(self.board if self.turn == P1 else np.flip(self.board))

    # Returns an int that uniquely identifies the state of this game. Useful for hashing
    def get_id(self):
        h = 0
        for r, c in np.ndindex(N, N):
            h = 3 * h + int(self.board[0, r, c]) - int(self.board[1, r, c])
        return h

    # Returns an array of bools denoting which moves are valid and invalid in the
    # current state from the current player's perspective
    def get_valid_moves(self):
        curr_player_board, opp_player_board = self.board if self.turn == P1 else np.flip(self.board)
        valid_moves = np.empty(TOTAL_MOVES, dtype=bool)
        valid_forward_moves = np.logical_and(
            curr_player_board[1:],
            np.logical_not(np.logical_or(curr_player_board[:-1], opp_player_board[:-1]))
        )
        valid_diag_left_moves = np.logical_and(curr_player_board[1:, 1:], np.logical_not(curr_player_board[:-1, :-1]))
        valid_diag_right_moves = np.logical_and(curr_player_board[1:, :-1], np.logical_not(curr_player_board[:-1, 1:]))
        valid_moves[:FORWARD_MOVES] = valid_forward_moves.reshape(-1)
        valid_moves[FORWARD_MOVES:-DIAG_MOVES] = valid_diag_left_moves.reshape(-1)
        valid_moves[-DIAG_MOVES:] = valid_diag_right_moves.reshape(-1)
        return valid_moves

    # Applies the given move to this game, mutating it
    def play_move(self, move):
        curr_player_board, opp_player_board = self.board if self.turn == P1 else np.flip(self.board)
        if move < FORWARD_MOVES:
            # Forward move
            r, c = divmod(move, N)
            curr_player_board[r, c] = PAWN
            curr_player_board[r + 1, c] = EMPTY
        elif move < FORWARD_MOVES + DIAG_MOVES:
            # Diagonal left move
            r, c = divmod(move - FORWARD_MOVES, N - 1)
            if self.turn == P1:
                self.p2_pawns -= int(opp_player_board[r, c])
            else:
                self.p1_pawns -= int(opp_player_board[r, c])
            curr_player_board[r, c] = PAWN
            curr_player_board[r + 1, c + 1] = EMPTY
            opp_player_board[r, c] = EMPTY
        else:
            # Diagonal right move
            r, c = divmod(move - FORWARD_MOVES - DIAG_MOVES, N - 1)
            if self.turn == P1:
                self.p2_pawns -= int(opp_player_board[r, c + 1])
            else:
                self.p1_pawns -= int(opp_player_board[r, c + 1])
            curr_player_board[r, c + 1] = PAWN
            curr_player_board[r + 1, c] = EMPTY
            opp_player_board[r, c + 1] = EMPTY
        if r == 0 or self.p1_pawns == 0 or self.p2_pawns == 0:
            self.winner = self.turn
        self.turn = -self.turn

    # Returns a new Breakthrough instance that is the result of playing the given move on this game
    def with_move(self, move):
        new_game = self.copy()
        new_game.play_move(move)
        return new_game

    def squares_to_move(self, src_square, dst_square):
        r1, c1 = src_square
        r2, c2 = dst_square
        if self.turn == P1:
            curr_player_board, opp_player_board = self.board
        else:
            curr_player_board, opp_player_board = np.flip(self.board)
            r1 = N - r1 - 1
            c1 = N - c1 - 1
            r2 = N - r2 - 1
            c2 = N - c2 - 1
        if not curr_player_board[r1, c1]:
            raise ValueError(f'Invalid move. Player has no pawn on source square.')
        if r2 - r1 != -1 or abs(c1 - c2) > 1:
            raise ValueError('Invalid move. Pawns can only move forwards straight ahead or diagonally 1 space.')
        if curr_player_board[r2, c2] or (c1 == c2 and opp_player_board[r2, c2]):
            raise ValueError(f'Invalid move. Destination square is blocked.')
        if c1 == c2:
            return r2 * N + c2
        elif c1 > c2:
            return FORWARD_MOVES + r2 * (N - 1) + c2
        else:
            return FORWARD_MOVES + DIAG_MOVES + r2 * (N - 1) + c2 - 1

    def is_game_over(self):
        return self.winner != 0

    # Returns 1 if player 1 won, or -1 if player 2 won. Breakthrough never ends in a draw
    def get_result(self):
        return self.winner

    def __str__(self):
        rows = []
        for r in range(N):
            row = [str(N - r) + ' ']
            for c in range(N):
                row.append('W' if self.board[0, r, c] else 'B' if self.board[1, r, c] else '.')
            rows.append(''.join(row))
        rows.append('  ' + ''.join(chr(ord('a') + i) for i in range(N)))
        return '\n'.join(rows)
