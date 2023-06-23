"""
Some self-defined wrappers for the MiniGrid environment.
"""

import gymnasium as gym


class AgentLocation(gym.core.Wrapper):
    """
    The wrapper to indicate the location of the agent in the `info`.
    """

    def __init__(self, env):
        super(AgentLocation, self).__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        agent_loc = self.unwrapped.agent_pos

        info['agent_loc'] = tuple(agent_loc)

        return observation, reward, terminated, truncated, info


class MovetoFourDirectionsWrapper(gym.core.Wrapper):
    """
    A wrapper to map from the original first three actions to four new actions:

    original actions:
    * 0 - turn left
    * 1 - turn right
    * 2 - move forward

    new mapped actions:
    * 0 - move forward
        2*
    * 1 - move to left
        0 2* 1
    * 2 - move to right
        1 2* 0
    * 3 - move backward
        0 0 2* 1 1

    Note: after stepping original action 2, need to record the `reward` and to check `done`.
    """

    def __init__(self, env):
        super(MovetoFourDirectionsWrapper, self).__init__(env)
        env.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        assert action in [0, 1, 2, 3], "{} is NOT an available action from MovetoFourDirectionsWrapper".format(action)

        # action 0 - move forward
        if action == 0:
            return self.env.step(2)

        # action 1 - move to left
        if action == 1:

            _, rewards, _, _, _ = self.env.step(0)

            o, r, te, tr, i = self.env.step(2)
            rewards += r

            self.unwrapped.step_count -= 1
            if te or tr:
                return o, rewards, te, tr, i

            o, r, te, tr, i = self.env.step(1)
            rewards += r

            self.unwrapped.step_count -= 1
            return o, rewards, te, tr, i

        # action 2 - move to right
        if action == 2:
            _, rewards, _, _, _ = self.env.step(1)

            o, r, te, tr, i = self.env.step(2)
            rewards += r

            self.unwrapped.step_count -= 1
            if te or tr:
                return o, rewards, te, tr, i

            o, r, te, tr, i = self.env.step(0)
            rewards += r

            self.unwrapped.step_count -= 1
            return o, rewards, te, tr, i

        # action 3 - move backward
        if action == 3:
            _, rewards, _, _, _ = self.env.step(0)

            _, r, _, _, _ = self.env.step(0)
            rewards += r

            o, r, te, tr, i = self.env.step(2)
            rewards += r

            self.unwrapped.step_count -= 2
            if te or tr:
                return o, rewards, te, tr, i

            _, r, _, _, _ = self.env.step(1)
            rewards += r

            _, r, _, _, _ = self.env.step(1)
            rewards += r

            self.unwrapped.step_count -= 2
            return o, rewards, te, tr, i
