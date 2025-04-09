class Arena:
    def __init__(self, game_cls):
        self.game_cls = game_cls

    def nnet_eval_play(self, mcts_player1, mcts_player2, temp):
        game = self.game_cls()
        while not game.is_game_over():
            move = mcts_player1.choose_best_move(game, temp) if game.turn == 1 else mcts_player2.choose_best_move(game, temp)
            game.play_move(move)
            mcts_player1.play_move(move)
            mcts_player2.play_move(move)
        return game.get_result()

    def play(self, player1, player2, game=None, display=False):
        if game is None:
            game = self.game_cls()
        if display:
            print(str(game), end='\n\n')
        while not game.is_game_over():
            move = player1.choose_best_move(game) if game.turn == 1 else player2.choose_best_move(game)
            game.play_move(move)
            if display:
                print(str(game), end='\n\n')
            player1.play_move(move)
            player2.play_move(move)
        if display:
            print(f'Player {"1" if game.get_result() == 1 else "2"} won the game')
        return game.get_result()
