class Arena:
    def __init__(self, game_cls):
        self.game_cls = game_cls

    def play(self, mcts_player1, mcts_player2, temp):
        game = self.game_cls()
        is_player1_turn = True
        while not game.is_game_over():
            move = mcts_player1.choose_best_move(game, temp) if is_player1_turn else mcts_player2.choose_best_move(game, temp)
            game.play_move(move)
            mcts_player1.play_move(move)
            mcts_player2.play_move(move)
            is_player1_turn = not is_player1_turn
        return game.get_result()
