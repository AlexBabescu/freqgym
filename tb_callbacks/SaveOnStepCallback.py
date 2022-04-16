from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


class SaveOnStepCallback(BaseCallback):

    def __init__(self, check_freq: int, save_name: str, save_dir: str, log_dir: str, verbose=0):
        super(SaveOnStepCallback, self).__init__(verbose)

        self.check_freq = check_freq
        self.log_dir = Path(log_dir)
        self.save_path = Path(save_dir) / save_name
        self.best_mean_reward = -np.inf
        self.best_net_worth = -np.inf

        assert self.log_dir.exists()
        assert Path(save_dir).exists()

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # total_profit = self.training_env.buf_infos[0]['total_profit']
        # self.logger.record('total_profit', total_profit)

        # total_reward = self.training_env.buf_infos[0]['total_reward']
        # self.logger.record('total_reward', total_reward)

        net_worth = self.training_env.buf_infos[0]['net_worth']
        self.logger.record('net_worth', net_worth)

        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(str(self.log_dir)), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, lets save
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)  # type: ignore

        return True

    def on_rollout_end(self):
        net_worth = self.training_env.buf_infos[0]['net_worth']
        if net_worth > self.best_net_worth:
            self.best_net_worth = net_worth
            save_name = f"{self.save_path}_net_worth"
            if self.verbose:
                print("Best net worth: {:.2f} - Last net worth: {:.2f}".format(self.best_net_worth, net_worth))
                print(f"Saving new best net worth to {save_name}")
            self.model.save(save_name)  # type: ignore
