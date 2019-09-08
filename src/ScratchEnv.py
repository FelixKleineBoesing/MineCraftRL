import minerl
import gym
from src.agents.ActorCritic import ActorCritic
from src.Helpers import ActionSpace


def main():
    env = gym.make('MineRLNavigateDense-v0')
    action_space = ActionSpace(action_space=env.action_space)

    agent = ActorCritic(action_space=action_space)
    state = env.reset()

    done = False
    while not done:
        action = agent.play_turn(state["pov"])
        #action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        #agent._get_feedback_inner(state, action, reward, next_state, done)
        state = next_state


if __name__ == '__main__':
    main()