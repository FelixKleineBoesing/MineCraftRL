import minerl
import gym


def main():
    env = gym.make('MineRLNavigateDense-v0')

    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    main()