import torch as th
import torch.nn as nn

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def to_gfn(module):
    module.log_Z = torch.zeros(1, requires_grad=True, device=device)


class Trajectories:
    def __init__(self, env_class, learning_module, batch_size):
        assert batch_size > 0

        self.envs = [env_class() for _ in range(batch_size)]
        map(lambda env: env.set_onehot(True), self.envs)

        # Remaining time = max trajectory length
        self.p1_last_states = th.zeros((batch_size, self.envs[0].observation_shape), device=device)
        self.p2_last_states = th.zeros((batch_size, self.envs[0].observation_shape), device=device)
        self.p1_actions = th.zeros((batch_size, self.envs[0].remaining_time), dtype=th.bool,
                                   device=device)
        self.p2_actions = th.zeros((batch_size, self.envs[0].remaining_time), dtype=th.bool,
                                   device=device)
        self.dones = th.zeros(batch_size, dtype=th.bool, device=device)

        self.learning_module = learning_module

        self.batch_size = batch_size

        # TODO find a better way to do this
        # Get initial states
        for i in range(batch_size):
            initial_states = self.envs[i].reset()[0]
            self.p1_last_states[i], self.p2_last_states[i] = \
                (th.tensor(initial_states["0"]["observation"], device=device),
                 th.tensor(initial_states["1"]["observation"], device=device))

    def step(self, other_module):
        assert not th.all(self.dones)

        # TODO should we compute p1 and p2 at the same time?
        p1_actions_log_probs = self.learning_module(self.p1_last_states)
        p2_actions_log_probs = self.other_module(self.p2_last_states)

        p1_chosen_actions = th.multinomial(p1_actions_log_probs.exp(), 1)
        p2_chosen_actions = th.multinomial(p2_actions_log_probs.exp(), 1)

        self.p1_actions[-self.envs[0].remaining_time] = p1_actions_log_probs[p1_chosen_actions]
        self.p2_actions[-self.envs[0].remaining_time] = p2_actions_log_probs[p2_chosen_actions]

        # TODO find a better way to do this ONLY WORK ON ENVS THAT ARE NOT DONE
        for i in th.arange(self.dones.shape[0])[~self.dones]:
        #     self.p1_last_states[i] = self.envs[i].step([p1_chosen_actions[i], p21_chosen_actions[i]])[0]

