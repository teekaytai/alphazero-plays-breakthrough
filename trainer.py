from collections import deque
import logging
import numpy as np
import os
import pickle
import random
import shutil
from tqdm import trange

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

logger = logging.getLogger()

class Trainer:
    def __init__(self, resume_from_latest=True):
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        self.arena = Arena(Breakthrough)
        if resume_from_latest:
            for d in os.listdir(CHECKPOINTS_DIR):
                directory = os.path.join(CHECKPOINTS_DIR, d)
                if os.path.isdir(directory) and d.startswith('latest'):
                    self.start_epoch = int(d.split('_')[-1]) + 1
                    with open(os.path.join(directory, TRAINING_DATA_FILE), 'rb') as f:
                        self.training_data_buffer = pickle.load(f)
                    model_path = os.path.join(directory, "model.pth")
                    self.nnet = Network(path=model_path)
                    logger.info(f'Resuming training from latest checkpoint')
                    return
        self.start_epoch = 1
        self.training_data_buffer = deque(maxlen=MAX_REPLAY_BUFFER_SIZE)
        self.nnet = Network()
        logger.info(f'Training new agent')

    def train(self, max_epochs):
        for epoch in range(self.start_epoch, max_epochs + 1):
            logger.info(f'Starting epoch {epoch}/{max_epochs}')
            logger.info('Running self-play games')
            for _ in trange(NUM_EPISODES):
                self.self_play_game(self.training_data_buffer, self.nnet)
            logger.info('Self-play games completed')

            logger.info('Training neural network')
            training_data = list(self.training_data_buffer)
            random.shuffle(training_data)
            new_nnet = self.nnet.copy()
            new_nnet.train(training_data)
            logger.info(f'New neural network trained')

            if self.is_new_nnet_better(new_nnet):
                logger.info('New neural network is better - replacing old network')
                self.nnet = new_nnet
            else:
                logger.info('New neural network is not better - keeping old network')
            self.update_checkpoints(epoch)
        logger.info('Agent training completed')

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
        logger.info('Running evaluation games')
        curr_mcts = MCTS(self.nnet)
        new_mcts = MCTS(new_nnet)
        curr_player = curr_mcts.choose_best_move
        new_player = new_mcts.choose_best_move
        new_player_wins = 0
        for i in trange(NUM_EVALUATION_GAMES):
            if i % 2 == 0:
                new_player_wins += self.arena.play(new_player, curr_player) == 1
            else:
                new_player_wins += self.arena.play(curr_player, new_player) == -1
        win_rate = new_player_wins / NUM_EVALUATION_GAMES
        logger.info(f'Evaluation games completed, win rate: {win_rate}')
        return win_rate >= WIN_THRESHOLD

    def update_checkpoints(self, epoch):
        logger.info('Saving checkpoint...')
        if epoch % CHECKPOINT_SAVE_FREQUENCY == 0:
            ckpt_dir = os.path.join(CHECKPOINTS_DIR, f'epoch_{epoch:04}')
            os.makedirs(ckpt_dir, exist_ok=True)
            self.save_checkpoint(ckpt_dir, epoch)
        latest_ckpt_dir = os.path.join(CHECKPOINTS_DIR, f'latest_epoch_{epoch:04}')
        os.makedirs(latest_ckpt_dir, exist_ok=True)
        self.save_checkpoint(latest_ckpt_dir)
        prev_latest_ckpt_dir = os.path.join(CHECKPOINTS_DIR, f'latest_epoch_{epoch - 1:04}')
        if os.path.exists(prev_latest_ckpt_dir):
            shutil.rmtree(prev_latest_ckpt_dir)
        logger.info('Saving complete')

    def save_checkpoint(self, directory, epoch=None):
        training_data_file = os.path.join(directory, 'training_data.pkl')
        with open(training_data_file, 'wb') as f:
            pickle.dump(self.training_data_buffer, f)
        self.nnet.save(path=os.path.join(directory, "model.pth"),
                   iteration=epoch)
