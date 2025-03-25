class Arena:
    def __init__(self, game_cls):
        self.game_cls = game_cls

    def play(self, player1, player2):
        game = self.game_cls()
        is_player1_turn = True
        while not game.is_game_over():
            move = player1(game) if is_player1_turn else player2(game)
            game.play_move(move)
            is_player1_turn = not is_player1_turn
        return game.get_result()
