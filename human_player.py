class HumanPlayer:
    def play_move(self, move):
        # No internal state to update, just do nothing
        pass

    def choose_best_move(self, game):
        n = game.board.shape[1]
        while True:
            curr_player = 'White' if game.turn == 1 else 'Black'
            input_str = input(f'Input move for {curr_player} (e.g. a1 a2): ')
            try:
                move_squares = self.parse_move(input_str, n)
                move = game.squares_to_move(*move_squares)
                print()
                return move
            except ValueError as e:
                print(e)

    def parse_move(self, input_str, n):
        square_strs = input_str.split()
        if len(square_strs) != 2 or not self.is_valid_square(square_strs[0], n) or not self.is_valid_square(square_strs[1], n):
            error_msg = ('Invalid move format. Input the move as two squares, ' +
                         'the square the pawn moves from and the square it moves to, separated by a space.')
            raise ValueError(error_msg)
        return self.parse_square(square_strs[0], n), self.parse_square(square_strs[1], n)

    def parse_square(self, square_str, n):
        return (n - int(square_str[1:])), ord(square_str[0]) - ord('a')

    def is_valid_square(self, square_str, n):
        if square_str[0] < 'a' or square_str[0] >= chr(ord('a') + n):
            return False
        return square_str[1:].isdigit() and 1 <= int(square_str[1:]) <= n
