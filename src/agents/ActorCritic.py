import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, LSTM, Input, Conv2D
import random
import os
import logging

from src.agents.Agent import Agent
from src.agents.ReplayBuffer import ReplayBuffer
from src.Helpers import multiply, min_max_scaling, ActionSpace


class ActorCritic(Agent):

    """
    this is a test agent which in the sense of simplicity predicts the binary actions but not the continous camera action
    """

    def __init__(self, action_space: ActionSpace,  epsilon: float = 0.0,
                 intervall_turns_train: int = np.Inf, intervall_turns_load: int = np.Inf,
                 save_path: str = "../data/modeldata/a2c/model.ckpt"):
        """
        Agent which implements Q Learning
        :param state_shape: shape of state
        :param action_shape: shape of actions
        :param name: name of agent
        :param side: is he going to start at the top or at the bottom?
        :param epsilon: exploration factor
        """

        #TODO add camera angle to actions
        self.number_actions = 512
        self.action_space = action_space

        # tensorflow related stuff
        self._batch_size = 512
        self._learning_rate = 0.01
        self._gamma = 0.99

        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        self.target_network = self._configure_network()
        self.network = self._configure_network()

        self.epsilon = epsilon
        self.exp_buffer = ReplayBuffer(100000)

        self._save_path = save_path
        if os.path.isfile(self._save_path + ".index"):
            self.network.load_weights(self._save_path)

        # copy weight to target weights
        self._load_weights_into_target_network()

        super().__init__()

    def decision(self, state_space: np.ndarray):
        """
        triggered by get play turn method of super class.
        This is the method were the magic should happen that chooses the right action
        :param state_space:
        :param action_space:
        :return:
        """
        # preprocess state space
        # normalizing state space between zero and one ( 2 is max value of stone and -2 is min value of stone
        state_space = min_max_scaling(state_space)

        qvalues, state_values = self.network([state_space])
        decision = self._sample_actions(qvalues)
        return decision

    def _sample_actions(self, qvalues: np.ndarray):
        """
        pick actions given qvalues. Uses epsilon-greedy exploration strategy.
        :param qvalues: output values from network
        :param action_space: action_space with all possible actions
        :return: return stone_id and move_id
        """
        epsilon = self.epsilon
        batch_size, x = qvalues.shape
        if random.random() < epsilon:
            decision = self.action_space.action_space.sample()
        else:
            possible_actions = qvalues_reshaped * action_space.space_array
            flattened_index = np.nanargmax(possible_actions)
            decision = np.array(np.unravel_index(flattened_index, self.action_shape))

        return decision

    def _get_feedback_inner(self, state, action, reward, next_state, finished):
        action_number = np.unravel_index(np.ravel_multi_index(action, self.action_shape), (512,))[0]
        self.exp_buffer.add(state, action_number, reward, next_state, finished)
        if self.number_turns % self._intervall_actions_train == 0 and self.number_turns > 1:
            self.train_network()
        if self.number_turns % self._intervall_turns_load == 0 and self.number_turns > 1:
            self._load_weights_into_target_network()

    def _load_weights_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        logging.debug("Transfer Weight!")
        self.network.save_weights(self._save_path)
        self.target_network.load_weights(self._save_path)

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        obs_batch = obs_batch.reshape(obs_batch.shape[0], 1, obs_batch.shape[1] * obs_batch.shape[2])
        next_obs_batch = next_obs_batch.reshape(next_obs_batch.shape[0], 1, next_obs_batch.shape[1] *
                                                next_obs_batch.shape[2])
        obs_batch = min_max_scaling(obs_batch)
        next_obs_batch = min_max_scaling(next_obs_batch)
        is_done_batch = is_done_batch.astype("float32")
        reward_batch = reward_batch.astype("float32")
        return {"obs": obs_batch, "actions": act_batch, "rewards": reward_batch,
                "next_obs": next_obs_batch, "is_done": is_done_batch}

    def train_network(self):
        logging.debug("Train Network!")
        loss_t = self._train_network(**self._sample_batch(batch_size=self._batch_size))
        self.td_loss_history.append(loss_t)
        self.moving_average_loss.append(np.mean([self.td_loss_history[max([0, len(self.td_loss_history) - 100]):]]))
        ma = self.moving_average_loss[-1]
        relative_ma = self.moving_average_loss[-1] / self._batch_size
        logging.info("Loss: {},     relative Loss: {}".format(ma, relative_ma))

    def _configure_network(self):
        # define network
        inputs = Input(shape=(64, 64, 3))
        x = Conv2D(32, (3,3), strides=2, activation="relu")(inputs)
        x = Conv2D(32, (3, 3), strides=2, activation="relu")(x)
        x = Conv2D(32, (3, 3), strides=2, activation="relu")(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)

        logits = Dense(512, activation="linear")(x)
        state_value = Dense(1, activation="linear")(x)
        network = tf.keras.models.Model(inputs=inputs, outputs=[logits, state_value])
        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        return network

    def _train_network(self, obs, actions, next_obs, rewards, is_done):

        # Decorator autographs the function
        @tf.function
        def td_loss():
            qvalues, state_values = self.network(obs)
            next_qvalues, next_state_values = self.target_network(next_obs)
            next_state_values = next_state_values * (1 - is_done)
            probs = tf.nn.softmax(qvalues)
            logprobs = tf.nn.log_softmax(qvalues)

            logp_actions = tf.reduce_sum(logprobs * tf.one_hot(actions, 1), axis=-1)
            advantage = rewards + self._gamma * next_state_values - state_values
            entropy = -tf.reduce_sum(probs * logprobs, 1, name="entropy")
            actor_loss = - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.001 * \
                         tf.reduce_mean(entropy)
            target_state_values = rewards + self._gamma * next_state_values
            critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values)) ** 2)
            return actor_loss + critic_loss

        with tf.GradientTape() as tape:
            loss = td_loss()

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        return loss
