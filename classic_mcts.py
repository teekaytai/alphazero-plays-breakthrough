import numpy as np
import collections
import math
from breakthrough import TOTAL_MOVES

# Exploration parameter for UCT
C = math.sqrt(2)

class ClassicMCTS:
    def __init__(self, num_iterations):
        self.root = None
        self.num_iterations = num_iterations

    def reset_tree(self):
        self.root = None

    def play_move(self, move):
        if self.root is None:
            return
        if move not in self.root.children:
            self.root.add_child(move)
        self.root = self.root.children[move]

    def choose_best_move(self, game):
        self.simulate(game)
        return np.argmax(self.root.child_N)

    def simulate(self, game):
        # Initialize root node
        if self.root is None:
            self.root = ClassicMCTSNode(game)

        for _ in range(self.num_iterations):
            selected = self.select_leaf()
            value = selected.play_out()
            selected.backup(value)

    def select_leaf(self):
        node = self.root
        while not node.game.is_game_over():
            best_move = node.best_child()
            if best_move not in node.children:
                return node.add_child(best_move)
            node = node.children[best_move]
        return node


# To maintain that every ClassicMCTSNode has a parent
class DummyNode:
    def __init__(self):
        self.parent = None
        self.child_W = collections.defaultdict(float)
        self.child_N = collections.defaultdict(float)

class ClassicMCTSNode:
    def __init__(self, game, move=None, parent=None):
        if parent is None:
            parent = DummyNode()

        self.game = game

        self.move = move
        self.parent = parent

        self.child_W = np.zeros(TOTAL_MOVES, dtype=np.float32)
        self.child_N = np.zeros(TOTAL_MOVES, dtype=np.float32)

        self.children = {} # maps moves to child nodes

    @property
    def N(self):
        return self.parent.child_N[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.move] = value

    @property
    def W(self):
        return self.parent.child_W[self.move]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.move] = value

    def child_Q(self):
        return self.child_W / (1.0 + self.child_N)

    def child_U(self):
        return C * np.sqrt(np.log(1.0 + self.N) / (1.0 + self.child_N))

    def child_score(self):
        # child_Q is from child's (other player's) perspective, negate to get curr player's perspective
        return -self.child_Q() + self.child_U()

    # Returns the best move based on uct score
    def best_child(self):
        valid_moves = self.game.get_valid_moves()
        scores = self.child_score()
        scores[~valid_moves] = -np.inf # eliminates illegal moves
        best_move = np.argmax(scores) # breakthrough should always have a valid move
        return best_move

    def backup(self, value):
        node = self
        while node.parent is not None:
            node.N += 1
            node.W += value
            node = node.parent
            value *= -1

    def add_child(self, move):
        new_game = self.game.with_move(move)
        self.children[move] = ClassicMCTSNode(new_game, move=move, parent=self)
        return self.children[move]

    def play_out(self):
        if self.game.is_game_over():
            return -1
        curr_game = self.game.copy()
        while not curr_game.is_game_over():
            valid_moves = curr_game.get_valid_moves()
            valid_moves_ids = np.where(valid_moves)[0]
            random_move = np.random.choice(valid_moves_ids)
            curr_game.play_move(random_move)
        return -1 if curr_game.turn == self.game.turn else 1
