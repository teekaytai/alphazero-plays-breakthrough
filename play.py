from arena import Arena
from breakthrough import Breakthrough
from human_player import HumanPlayer
from mcts import MCTS
from models.model_2.breakthrough_net import BreakThroughNet as bnnet
from network import Network

if __name__ == '__main__':
    arena = Arena(Breakthrough)
    human_player = HumanPlayer()
    network = Network(bnnet, 'models/model_2/model.pth')
    mcts_player = MCTS(network)
    _input = input('Would you like to go first? (y/n): ').lower()
    while _input not in ('y', 'yes', 'n', 'no'):
        _input = input('Would you like to go first? (y/n): ').lower()
    print()
    if _input in ('y', 'yes'):
        arena.play(human_player, mcts_player, display=True)
    else:
        arena.play(mcts_player, human_player, display=True)
