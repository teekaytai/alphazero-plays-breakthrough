import numpy as np
from tqdm import tqdm, trange

from arena import Arena
from breakthrough import Breakthrough
from classic_mcts import ClassicMCTS
from mcts import MCTS
from models.model_2.breakthrough_net import BreakThroughNet as bnnet2
from network import Network

# Plays two MCTS players against each other on all possible games after the first move
# and returns the number of games each player won.
def evaluate(mcts1, mcts2):
    arena = Arena(Breakthrough)
    start_game = Breakthrough()
    first_move_ids = np.where(start_game.get_valid_moves())[0]
    total_games = 2 * len(first_move_ids)
    mcts1_wins = 0
    for first_move in tqdm(first_move_ids):
        game = start_game.with_move(first_move)
        mcts1.reset_tree()
        mcts2.reset_tree()
        mcts1_wins += arena.play(mcts1, mcts2, game.copy()) == 1
        mcts1.reset_tree()
        mcts2.reset_tree()
        mcts1_wins += arena.play(mcts2, mcts1, game) == -1
    return mcts1_wins, total_games - mcts1_wins

# Plays two MCTS players against each other for a given number of games
# and returns the number of games each player won.
def evaluate2(mcts1, mcts2, total_games=30):
    arena = Arena(Breakthrough)
    mcts1_wins = 0
    for i in trange(total_games):
        mcts1.reset_tree()
        mcts2.reset_tree()
        if i % 2 == 0:
            mcts1_wins += arena.play(mcts1, mcts2) == 1
        else:
            mcts1_wins += arena.play(mcts2, mcts1) == -1
    return mcts1_wins, total_games - mcts1_wins

if __name__ == '__main__':
    classic_mcts = ClassicMCTS(10000)
    network2 = Network(bnnet2, 'models/model_2/model.pth')
    mcts2 = MCTS(network2)
    print(evaluate2(mcts2, classic_mcts))
