import numpy as np
from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None
        self.observation_shape = (1, 1, 1)
        self.action_space = [0, 1]
        self.players = [0]
        self.stacked_observations = 0
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 200
        self.num_simulations = 10
        self.discount = 0.997
        self.temperature_threshold = None
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.network = "fullyconnected"
        self.support_size = 5
        self.encoding_size = 8
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [16]
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]
        import pathlib
        self.results_path = pathlib.Path(__file__).resolve().parent / "results"
        self.save_model = True
        self.training_steps = 100
        self.batch_size = 32
        self.checkpoint_interval = 10
        self.value_loss_weight = 1
        self.train_on_gpu = False
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = 0.02
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 100
        self.replay_buffer_size = 50
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.PER = True
        self.PER_alpha = 0.5
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1.5

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        return 1.0 if trained_steps < 50 else 0.5

class Game(AbstractGame):
    def __init__(self, seed=None, data_file="stock_prices.csv"):
        import pandas as pd
        self.data = pd.read_csv(data_file)
        self.prices = self.data["Close"].values
        self.index = 0

    def step(self, action):
        if self.index >= len(self.prices) - 2:
            return self._get_obs(), 0, True
        current = self.prices[self.index]
        next_price = self.prices[self.index + 1]
        diff = next_price - current
        reward = 1 if (diff > 0 and action == 1) or (diff <= 0 and action == 0) else -1
        self.index += 1
        done = self.index >= len(self.prices) - 1
        return self._get_obs(), reward, done

    def legal_actions(self):
        return [0, 1]

    def reset(self):
        self.index = 0
        return self._get_obs()

    def _get_obs(self):
        val = self.prices[self.index] if self.index < len(self.prices) else 0.0
        return np.array([[val]])

    def close(self):
        pass

    def render(self):
        pass

    def action_to_string(self, action_number):
        return {0: "Predict Down", 1: "Predict Up"}[action_number]
