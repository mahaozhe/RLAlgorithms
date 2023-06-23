import gymnasium as gym


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
            _, rewards, _, _ = self.env.step(0)

            s, r, d, i = self.env.step(2)
            rewards += r
            if d:
                return s, rewards, d, i

            _, r, _, _ = self.env.step(1)
            rewards += r

            return s, rewards, d, i

        # action 2 - move to right
        if action == 2:
            _, rewards, _, _ = self.env.step(1)

            s, r, d, i = self.env.step(2)
            rewards += r
            if d:
                return s, rewards, d, i

            _, r, _, _ = self.env.step(0)
            rewards += r

            return s, rewards, d, i

        # action 3 - move backward
        if action == 3:
            _, rewards, _, _ = self.env.step(0)

            _, r, _, _ = self.env.step(0)
            rewards += r

            s, r, d, i = self.env.step(2)
            rewards += r
            if d:
                return s, rewards, d, i

            _, r, _, _ = self.env.step(1)
            rewards += r

            _, r, _, _ = self.env.step(1)
            rewards += r

            return s, rewards, d, i


class AgentCoordinates(gym.core.Wrapper):
    """
    The wrapper to indicate the location of the agent.
    """

    def __init__(self, env):
        super(AgentCoordinates, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        agent_loc = self.unwrapped.agent_pos

        info['agent_loc'] = tuple(agent_loc)

        return obs, reward, done, info


class AutoPickUpKeyOpenDoor(gym.core.Wrapper):
    """
    The wrapper to make the agent to automatically pick up the key once it steps on it.
    Once the agent picks up the key, the door will be set to open for it to pass through.
    """

    def __init__(self, env):
        super(AutoPickUpKeyOpenDoor, self).__init__(env)

        self.key_num = len(env.unwrapped.key_locs)

        env.unwrapped.key_picked = [False] * self.key_num

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.unwrapped.key_picked = [False] * self.key_num

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped

        agent_loc = tuple(self.unwrapped.agent_pos)

        key_loc = env.key_locs
        door_loc = env.door_locs

        if agent_loc in key_loc:
            # auto pick up the key
            key_index = key_loc.index(agent_loc)
            if not env.key_picked[key_index]:
                self.grid.set(key_loc[key_index][0], key_loc[key_index][1], None)
                door_obj = env.grid.get(door_loc[key_index][0], door_loc[key_index][1])
                door_obj.is_locked = False
                door_obj.is_open = True
                env.key_picked[key_index] = True

        return obs, reward, done, info


class KeyDoorSubGoalsIndicator(gym.core.Wrapper):
    """
    The wrapper to return info to indicate the key-picking and door-opening.
    """

    def __init__(self, env):
        super(KeyDoorSubGoalsIndicator, self).__init__(env)

        self.key_num = len(env.unwrapped.key_locs)

    def reset(self, **kwargs):
        obs = self.env.reset()

        # + set the flag to make sure the goal can be completed only once.
        self.unwrapped.key_picked_r = [False] * self.key_num
        self.unwrapped.door_has_opened_r = [False] * self.key_num

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped

        info['key_picked'] = [False] * len(env.key_locs)
        info['door_passed'] = [False] * len(env.key_locs)

        agent_loc = tuple(self.unwrapped.agent_pos)

        key_loc = env.key_locs
        door_loc = env.door_locs

        # * check whether the key is picked up
        if agent_loc in key_loc:
            key_index = key_loc.index(agent_loc)
            if not env.key_picked_r[key_index]:
                info['key_picked'][key_index] = True
                env.key_picked_r[key_index] = True
                # / set that the door has been opened
                env.door_has_opened_r[key_index] = True

        # * check whether the door is passed, which can be completed more times.
        if agent_loc in door_loc:
            door_index = door_loc.index(agent_loc)
            info['door_passed'][door_index] = True

        info['door_has_opened'] = env.door_has_opened_r

        return obs, reward, done, info
