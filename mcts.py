import numpy as np
import collections
import math
from breakthrough import TOTAL_MOVES

NUM_ITERATIONS = 1000 # Default number of MCTS simulations per move
C_BASE = 19652 # Determines how quickly exploration bonus decreases as visits accumulate
C_INIT = 1.25 # Initial exploration weight before visits accumulate
class MCTS:
    def __init__(self, nnet, c_base = C_BASE, c_init = C_INIT):
        self.nnet = nnet
        self.root = None
        self.c_base = c_base
        self.c_init = c_init
        self.nn_cache = {}

    # Returns a vector describing the probabilities of choosing each move in the given game
    def compute_policy(self, game, num_iterations = NUM_ITERATIONS):
        self.simulate(game, num_iterations)
        visit_counts = self.root.child_N
        return visit_counts / self.N

    # Returns the best move in the given game
    def choose_best_move(self, game, num_iterations = NUM_ITERATIONS):
        self.simulate(game, num_iterations)
        visit_counts = self.root.child_N
        return np.argmax(visit_counts)

    # Runs MCTS for the given number of iterations
    def simulate(self, game, num_iterations):
        terminal_reward = 1

        # Initialize root node
        self.root = MCTSNode(game)
        policy, value = self.cached_predict(game)
        self.root.expand(policy)
        self.root.backup(value)

        for _ in range(num_iterations):
            selected = self.select_leaf()
            
            if selected.game.is_game_over():
                selected.backup(-terminal_reward) # prev player won
            else:
                policy, value = self.cached_predict(selected.game)
                selected.expand(policy)
                selected.backup(value)

    def select_leaf(self):
        node = self.root
        while node.is_expanded:
            best_move = node.best_child(self.c_init, self.c_base)
            node = node.try_add_child(best_move)
        return node
    
    def cached_predict(self, game):
        game_state = game.get_state()
        game_state_key = game_state.tobytes()

        cached_result = self.nn_cache.get(game_state_key)
        if cached_result is not None:
            return cached_result
        
        result = self.nnet.predict(game_state)
        self.nn_cache[game_state_key] = result
        return result


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
    
    def child_U(self, c_init, c_base):
        c_puct = c_init + math.log((self.N + c_base + 1) / c_base)
        return c_puct * self.child_P * math.sqrt(self.N) / (1.0 + self.child_N)

    def child_score(self, c_init, c_base):
        # child_Q is from child's (other player's) perspective, negate to get curr player's perspective
        return -self.child_Q() + self.child_U(c_init, c_base)

    # Returns the best move based on uct score
    def best_child(self, c_init, c_base):
        valid_moves = self.game.get_valid_moves()
        scores = self.child_score(c_init, c_base)
        scores[~valid_moves] = -np.inf # eliminates illegal moves
        best_move = np.argmax(scores) # breakthrough should always have a valid move
        return best_move


    # Expands the current node (just fills up prior probabilities)
    def expand(self, child_P):
        self.child_P = child_P
        self.is_expanded = True
    
    def backup(self, value):
        node = self
        while node.parent is not None:
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