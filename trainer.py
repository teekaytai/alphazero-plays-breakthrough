from collections import deque
import numpy as np
import os
import pickle
import random
import shutil

from arena import Arena
from breakthrough import Breakthrough
from mcts import MCTS
from network import Network

CHECKPOINTS_DIR = 'checkpoints'
TRAINING_DATA_FILE = 'training_data.pkl'
CHECKPOINT_SAVE_FREQUENCY = 10
MAX_REPLAY_BUFFER_SIZE = 500000
NUM_EPISODES = 100
NUM_EVALUATION_GAMES = 50
WIN_THRESHOLD = 0.55

class Trainer:
    def __init__(self, resume_from_latest=True):
        self.arena = Arena(Breakthrough)
        if resume_from_latest:
            for d in os.listdir(CHECKPOINTS_DIR):
                directory = os.path.join(CHECKPOINTS_DIR, d)
                if os.path.isdir(directory) and d.startswith('latest'):
                    self.start_epoch = int(d.split('_')[-1]) + 1
                    self.training_data_buffer = pickle.loads(os.path.join(directory, TRAINING_DATA_FILE))
                    self.nnet = Network(directory)
                    return
        self.start_epoch = 1
        self.training_data_buffer = deque(maxlen=MAX_REPLAY_BUFFER_SIZE)
        self.nnet = Network()

    def train(self, max_epochs):
        for epoch in range(self.start_epoch, max_epochs + 1):
            for _ in range(NUM_EPISODES):
                self.self_play_game(self.training_data_buffer, self.nnet)
            training_data = list(self.training_data_buffer)
            random.shuffle(training_data)
            new_nnet = self.nnet.copy()
            new_nnet.train(training_data)
            if self.is_new_nnet_better(new_nnet):
                self.nnet = new_nnet
            self.update_checkpoints(epoch)

    def self_play_game(self, training_data_buffer, nnet):
        num_possible_moves = Breakthrough.num_possible_moves()
        game = Breakthrough()
        mcts = MCTS(nnet)
        states = []
        policies = []
        while not game.is_game_over():
            policy = mcts.compute_policy(game)
            states.append(game.get_state())
            policies.append(policy)
            chosen_move = np.random.choice(num_possible_moves, p=policy)
            game.play_move(chosen_move)
        result = game.get_result()
        for i, (state, policy) in enumerate(zip(states, policies)):
            value_target = result if i % 2 == 0 else -result
            training_data_buffer.append((state, policy, value_target))

    def is_new_nnet_better(self, new_nnet):
        curr_mcts = MCTS(self.nnet)
        new_mcts = MCTS(new_nnet)
        curr_player = curr_mcts.choose_best_move
        new_player = new_mcts.choose_best_move
        new_player_wins = 0
        for i in range(NUM_EVALUATION_GAMES):
            if i % 2 == 0:
                new_player_wins += self.arena.play(new_player, curr_player) == 1
            else:
                new_player_wins += self.arena.play(curr_player, new_player) == -1
        win_rate = new_player_wins / NUM_EVALUATION_GAMES
        return win_rate >= WIN_THRESHOLD

    def update_checkpoints(self, epoch):
        if epoch % CHECKPOINT_SAVE_FREQUENCY == 0:
            ckpt_dir = os.path.join(CHECKPOINTS_DIR, f'epoch_{epoch:04}')
            self.save_checkpoint(ckpt_dir)
        latest_ckpt_dir = os.path.join(CHECKPOINTS_DIR, f'latest_epoch_{epoch:04}')
        self.save_checkpoint(latest_ckpt_dir)
        prev_latest_ckpt_dir = os.path.join(CHECKPOINTS_DIR, f'latest_epoch_{epoch - 1:04}')
        if os.path.exists(prev_latest_ckpt_dir):
            shutil.rmtree(prev_latest_ckpt_dir)

    def save_checkpoint(self, training_data_buffer, nnet, directory):
        training_data_file = os.path.join(directory, 'training_data.pkl')
        with open(training_data_file, 'wb') as f:
            pickle.dump(training_data_buffer, f)
        nnet.save(directory)
