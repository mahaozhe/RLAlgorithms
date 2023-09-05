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
    The wrapper to modify the action space to `Discrete(4)`,
    making the agent only moves to four directions for one step:

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


class AutoPickUpKeyOpenDoor(gym.core.Wrapper):
    """
    The wrapper to make the agent to automatically pick up the key once it steps on it.
    Once the agent picks up the key, the door will be set to open for it to pass through.
    """

    def __init__(self, env):
        super(AutoPickUpKeyOpenDoor, self).__init__(env)

        self.key_num = len(env.unwrapped.key_locs)
        self.key_picked = [False] * self.key_num

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.key_picked = [False] * self.key_num

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped

        agent_loc = env.agent_pos

        # get the key and the door locations
        key_locs = env.key_locs
        door_locs = env.door_locs

        # check if the agent steps on the key, if so, auto-pick up the key, and open the door
        for i in range(self.key_num):
            if not self.key_picked[i] and agent_loc == key_locs[i]:
                self.key_picked[i] = True
                self.grid.set(key_locs[i][0], key_locs[i][1], None)
                door_obj = env.grid.get(door_locs[i][0], door_locs[i][1])
                door_obj.is_locked = False
                door_obj.is_open = True
