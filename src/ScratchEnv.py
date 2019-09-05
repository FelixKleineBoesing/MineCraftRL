import minerl
import gym

from src.agents.QLearningAgent import QLearningAgent


def main():
    env = gym.make('MineRLNavigateDense-v0')
    agent = QLearningAgent()
    state = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        agent._get_feedback_inner(state, action, reward, next_state, done)
        state = next_state


if __name__ == '__main__':
    main()