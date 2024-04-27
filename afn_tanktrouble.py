from pettingzoo.utils import parallel_to_aec
import numpy as np
from torch import optim
import wandb

from tanktrouble_gym.tanktrouble.env.tanktrouble_env import TankTrouble

from src.data import TrajectoryBuffer
from src.envs.base_env import BaseEnv, Outcome, Player
from src.models.mlp import MLP
from src.tb import train

class AttrDict(dict):
    def __getattr__(self, k):
        return self[k]


class TankTroubleAFN(BaseEnv):
    def __init__(
        self,
        s_x: int = 8,
        s_y: int = 5,
        lambd=15,
        seed: int = None, 
    ) -> None:
        self.name = "TankTrouble"
        self.num_rows = s_y
        self.num_cols = s_x

        self._env = TankTrouble(s_x, s_y)
        self._env.set_onehot(True)
        self.seed = seed

        self.BOARD_SHAPE = self._env.observation_shape
        self.STATE_DIM = (self._env.observation_shape,)
        self.NUM_EXTRA_INFO = None
        self.ACTION_DIM = self._env.action_shape

        self._env = parallel_to_aec(self._env)

        self.turns = 0

        self.reset()

        self.MAX_TRAJ_LEN = 2 * self._env.env.env.remaining_time + 1

    def reset(self) -> None:
        self._env.reset(seed=self.seed)
        self.done = False
        self.outcome = Outcome.NOT_DONE
        self.curr_player = Player.ONE

    def place_piece(self, action: int) -> None:
        # print("agent action")
        self._env.step(action)

    def get_extra_info(self) -> np.ndarray:
        return None

    def get_masks(self) -> np.ndarray:
        return self._env.env.env.mask
    
    def obs(self):
        obs, _, _, _, _ = self._env.last()
        return obs["observation"]

    def evaluate_outcome(self) -> Outcome:
        rewards = self._env.rewards
        _, _, done, _, _ = self._env.last()

        if rewards[str(Player.ONE)] > 1:
            return Outcome.WIN_P1
        elif rewards[str(Player.TWO)] > 1:
            return Outcome.WIN_P2
        elif done:
            return Outcome.DRAW
        else:
            return Outcome.NOT_DONE

    def render(self) -> None:
        # self._env.render()
        pass



def main():
    env = TankTroubleAFN()
    model = MLP(env.BOARD_SHAPE, env.ACTION_DIM)
    optimizer = optim.Adam([{"params": model.parameters(), "lr": 1e-3}, {"params": model.log_Z, "lr": 5e-2}])
    buffer = TrajectoryBuffer(2, env)
    train_cfg = AttrDict({
        "batch_size": 64,
        "total_steps": 20_000,
        "eval_every": 100,
        "buffer_batch_size": 64,
        "num_initial_traj": 64,
        "num_regen_traj": 64,
        "regen_every": 64,
        "ckpt_dir": "~/scratch/checkpoints-TankTrouble"
    })

    train_cfg.__getattribute__ = lambda self, x : self[x]

    wandb.init(project="AFN", tags=["TankTrouble"])

    train(env, model, optimizer, buffer, train_cfg)

if __name__ == "__main__":
    main()

