import gymnasium as gym

from RLEnvs.MyMiniWorld import collecthealth, pickupobjects

# env_name = "MyMiniWorld/MiniWorld-CollectHealth"
#
# # env = gym.make(env_name)  # no render
# # env = gym.make(env_name, render_mode="human")  # auto render, no return from env.render()
# env = gym.make(env_name, render_mode="rgb_array")  # return a (x,y,3) np.ndarray by env.render()
#
# observation, info = env.reset(seed=0)
#
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     rendered = env.render()   # if `render_mode` is "rgb_array", return a rendered image.
#
#     if terminated or truncated:
#         observation, info = env.reset()
#
# env.close()

import math
import argparse

import pyglet
from pyglet.window import key


class ManualControl:
    def __init__(self, env, no_time_limit, domain_rand):
        self.env = env

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
        print("============")

        self.env.reset()

        # Create the display window
        self.env.render()

        env = self.env

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self.env.close()

            if symbol == key.UP:
                self.step(self.env.actions.move_forward)
            elif symbol == key.DOWN:
                self.step(self.env.actions.move_back)
            elif symbol == key.LEFT:
                self.step(self.env.actions.turn_left)
            elif symbol == key.RIGHT:
                self.step(self.env.actions.turn_right)
            elif symbol == key.PAGEUP or symbol == key.P:
                self.step(self.env.actions.pickup)
            elif symbol == key.PAGEDOWN or symbol == key.D:
                self.step(self.env.actions.drop)
            elif symbol == key.ENTER:
                self.step(self.env.actions.done)

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            pass

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            pyglet.app.exit()

        # Enter main event loop
        pyglet.app.run()

        self.env.close()

    def step(self, action):
        print(
            "step {}/{}: {}".format(
                self.env.step_count + 1,
                self.env.max_episode_steps,
                self.env.actions(action).name,
            )
        )

        obs, reward, termination, truncation, info = self.env.step(action)

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            self.env.reset()

        self.env.render()


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env-name", default="MyMiniWorld/MiniWorld-CollectHealth")
    # parser.add_argument("--env-name", default="MyMiniWorld/MiniWorld-PickupObjects-Pos-1-10")
    # parser.add_argument("--env-name", default="MyMiniWorld/MiniWorld-PickupObjects-Pos-10-1")
    # parser.add_argument("--env-name", default="MyMiniWorld/MiniWorld-PickupObjects-Pos-10-10")
    parser.add_argument("--env-name", default="MyMiniWorld/MiniWorld-PickupObjects-Pos-5-10")
    # parser.add_argument("--env-name", default="MyMiniWorld/MiniWorld-PickupObjects-Pos-10-5")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env = gym.make(args.env_name, view=view_mode, render_mode="human")

    manual_control = ManualControl(env, args.no_time_limit, args.domain_rand)
    manual_control.run()


if __name__ == "__main__":
    main()