import numpy as np
import collections
import math
from breakthrough import TOTAL_MOVES, Breakthrough

class MCTS:
    def __init__(self, nnet, c_puct = 1.0, num_iterations = 100):
        self.nnet = nnet
        self.root: MCTSNode = None
        self.c_puct = c_puct
        self.num_iterations = num_iterations

    # Returns a vector describing the probabilities of choosing each move in the given game
    def compute_policy(self, game):
        self.simulate(game, self.num_iterations)
        visit_counts = self.root.child_N
        return visit_counts / np.sum(visit_counts)

    # Returns the best move in the given game
    def choose_best_move(self, game):
        self.simulate(game, self.num_iterations)
        visit_counts = self.root.child_N
        return np.argmax(visit_counts)

    # Runs MCTS for the given number of iterations
    def simulate(self, game, num_iterations):
        terminal_reward = 1

        # Initialize root node
        self.root = MCTSNode(game)
        policy, value = self.nnet.predict(self.root.game.get_state())
        self.root.expand(policy)
        self.root.backup(value)

        for _ in range(num_iterations):
            selected = self.select_leaf()
            
            if selected.game.is_game_over():
                selected.backup(-terminal_reward)
            else:
                policy, value = self.nnet.predict(selected.game.get_state())
                selected.backup(value)

    def select_leaf(self):
        node = self.root
        while node.is_expanded:
            best_move = node.best_child(self.c_puct)
            node = node.try_add_child(best_move)
        # Break out when node is a new child
        return node

# To maintain that every MCTSNode has a parent
class DummyNode:
    def __init__(self):
        self.parent = None
        self.child_W = collections.defaultdict(float)
        self.child_N = collections.defaultdict(float)

class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        if parent is None:
            parent = DummyNode()

        self.game = game

        self.move = move
        self.parent = parent
        self.is_expanded = False
        
        self.child_W = np.zeros(TOTAL_MOVES, dtype=np.float32)
        self.child_N = np.zeros(TOTAL_MOVES, dtype=np.float32)
        self.child_P = np.zeros(TOTAL_MOVES, dtype=np.float32)

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
    
    def child_U(self, c_puct):
        return c_puct * self.child_P * math.sqrt(self.N) / (1.0 + self.child_N)

    def child_score(self, c_puct):
        # child_Q is from child's (other player's) perspective, negate to get curr player's perspective
        return -self.child_Q() + self.child_U(c_puct)

    # Returns the best move based on uct score
    def best_child(self, c_puct):
        valid_moves = self.game.get_valid_moves()
        scores = self.child_score(c_puct)
        scores[~valid_moves] = -np.inf # eliminates illegal moves
        best_move = np.argmax(scores)
        return best_move


    # Expands the current node (just fills up prior probabilities)
    def expand(self, child_P):
        self.child_P = child_P
        self.is_expanded = True
    
    def backup(self, value):
        node = self
        while isinstance(node, MCTSNode):
            node.N += 1
            node.W += value
            node = node.parent
            value *= -1

    def try_add_child(self, move):
        if move in self.children:
            return self.children[move]

        new_game = self.game.with_move(move)
        self.children[move] = MCTSNode(new_game, move=move, parent=self)
        return self.children[move]