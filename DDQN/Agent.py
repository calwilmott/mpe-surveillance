import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
from tensorflow.keras.utils import plot_model
import numpy as np


class DDQNAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5
        self.alpha = 0.005
        self.gamma = 0.95

        # Exploration strategy
        self.soft_max_scaling = 0.1

        # Global-Local Map
        self.use_global_local = True
        self.global_map_scaling = 3
        self.local_map_size = 17

        # Printing
        self.print_summary = False


class DDQNAgent(object):

    def __init__(self, params: DDQNAgentParams, example_state, action_space, stats=None, observation_mode="hybrid",
                 num_agents=3, deep_discretization=False):
        self.params = params
        self.obs_mode = observation_mode
        self.num_agents = num_agents
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0
        self.deep_discretization = deep_discretization

        if self.obs_mode == "image":
            grid_dim = np.shape(example_state)[0]
            self.boolean_map_shape = (grid_dim, grid_dim, np.shape(example_state)[2] - 1)
            self.float_map_shape = (grid_dim, grid_dim, 1)
            self.scalars = None
        elif self.obs_mode == "hybrid":
            image_obs = example_state["image"]
            grid_dim = np.shape(image_obs)[0]
            self.boolean_map_shape = (grid_dim, grid_dim, np.shape(image_obs)[2] - 1)
            self.float_map_shape = (grid_dim, grid_dim, 1)
            dense_obs = example_state["dense"]
            self.scalars = np.shape(dense_obs)[0]
        else:
            self.boolean_map_shape = example_state.get_boolean_map_shape()
            self.float_map_shape = example_state.get_float_map_shape()
            self.scalars = example_state.get_num_scalars()

        self.num_actions = action_space
        if self.obs_mode == "hybrid":
            self.num_map_channels = image_obs[2]
        else:
            self.num_map_channels = example_state[2]

        # Create shared inputs
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        states = [float_map_input]

        if self.boolean_map_shape is not None:
            boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
            states.append(boolean_map_input)
            map_cast = tf.cast(boolean_map_input, dtype=tf.float32)
            padded_map = tf.concat([map_cast, float_map_input], axis=3)
        else:
            boolean_map_input = None
            padded_map = tf.concat([float_map_input], axis=3)
        if self.scalars is not None:
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states.append(scalars_input)
        else:
            scalars_input = None
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_star_input = Input(shape=(), name='q_star_input', dtype=tf.float32)

        self.q_network = self.build_model(padded_map, scalars_input, states)
        self.target_network = self.build_model(padded_map, scalars_input, states, 'target_')
        self.hard_update()

        if self.params.use_global_local:
            if boolean_map_input is None:
                self.global_map_model = Model(inputs=[float_map_input], outputs=self.global_map)
                self.local_map_model = Model(inputs=[float_map_input], outputs=self.local_map)
                self.total_map_model = Model(inputs=[float_map_input], outputs=self.total_map)
            else:
                self.global_map_model = Model(inputs=[boolean_map_input, float_map_input], outputs=self.global_map)
                self.local_map_model = Model(inputs=[boolean_map_input, float_map_input], outputs=self.local_map)
                self.total_map_model = Model(inputs=[boolean_map_input, float_map_input], outputs=self.total_map)

        q_values = self.q_network.output
        q_target_values = self.target_network.output

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action = tf.argmax(q_values, axis=1, name='max_action', output_type=tf.int64)
        max_action_target = tf.argmax(q_target_values, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=float)
        q_star = tf.reduce_sum(tf.multiply(one_hot_max_action, q_target_values, name='mul_hot_target'), axis=1,
                               name='q_star')
        self.q_star_model = Model(inputs=states, outputs=q_star)

        # Define Bellman loss
        max_action_input = tf.argmax(action_input, axis=1, name='max_action_input', output_type=tf.int64)
        one_hot_rm_action = tf.one_hot(max_action_input, depth=self.num_actions, on_value=1.0, off_value=0.0, dtype=float)
        one_cold_rm_action = tf.one_hot(max_action_input, depth=self.num_actions, on_value=0.0, off_value=1.0, dtype=float)
        q_old = tf.stop_gradient(tf.multiply(q_values, one_cold_rm_action))
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated)), 1)
        q_update_hot = tf.multiply(q_update, one_hot_rm_action)
        q_new = tf.add(q_update_hot, q_old)
        q_loss = tf.losses.MeanSquaredError()(q_new, q_values)

        q_loss_model_inputs = [float_map_input]
        if boolean_map_input is not None:
            q_loss_model_inputs.append(boolean_map_input)
        if scalars_input is not None:
            q_loss_model_inputs.append(scalars_input)
        q_loss_model_inputs.extend([action_input, reward_input, termination_input, q_star_input])
        self.q_loss_model = Model(inputs=q_loss_model_inputs, outputs=q_loss)

        # Exploit act model
        self.exploit_model = Model(inputs=states, outputs=max_action)
        self.exploit_model_target = Model(inputs=states, outputs=max_action_target)

        # Softmax explore model
        softmax_scaling = tf.divide(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        self.soft_explore_model = Model(inputs=states, outputs=softmax_action)

        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            plot_model(self.q_loss_model, to_file="model.png", show_shapes=True)
            self.q_loss_model.summary()

        if stats:
            stats.set_model(self.target_network)

    def build_model(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)

        if states_proc is not None:
            layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        else:
            layer = Concatenate(name=name + 'concat')([flatten_map])

        for k in range(self.params.hidden_layer_num):
            hidden_layer_size = self.params.hidden_layer_size
            layer = Dense(hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def create_map_proc(self, conv_in, name):
        if self.params.use_global_local:
            # Forking for global and local map
            # Global Map
            global_map = tf.stop_gradient(
                AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

            self.global_map = global_map
            self.total_map = conv_in

            for k in range(self.params.conv_layers):
                global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                    strides=(1, 1),
                                    name=name + 'global_conv_' + str(k + 1))(global_map)

            flatten_global = Flatten(name=name + 'global_flatten')(global_map)

            # Local Map
            crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
            local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
            self.local_map = local_map

            for k in range(self.params.conv_layers):
                local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                   strides=(1, 1),
                                   name=name + 'local_conv_' + str(k + 1))(local_map)

            flatten_local = Flatten(name=name + 'local_flatten')(local_map)

            return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])
        else:
            conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu', strides=(1, 1),
                              name=name + 'map_conv_0')(conv_in)
            for k in range(self.params.conv_layers - 1):
                conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                  strides=(1, 1),
                                  name=name + 'map_conv_' + str(k + 1))(conv_map)

            flatten_map = Flatten(name=name + 'flatten')(conv_map)
            return flatten_map

    def one_hot_action(self, action_nums):
        action_array = np.zeros((self.num_agents, self.num_actions))
        for i in range(len(action_nums)):
            if not self.deep_discretization:
                action_array[i][action_nums[i]] = 1.0
            else:
                first_action = action_nums[i] % self.num_actions
                action_array[i][first_action] = 1.0
                for j in range(1, 4):
                    action = action_nums[i] // int(pow(self.num_actions, j))
                    action_array[i][action] = 1.0
        return action_array

    def get_random_action(self):
        action_nums = np.random.randint(0, high=self.num_actions, size=self.num_agents)
        action_array = self.one_hot_action(action_nums)
        return action_array

    def get_exploitation_action(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def _stack_along(self, previous, new, iteration):
        if iteration == 0:
            return new
        elif iteration == 1:
            return tf.stack([previous, new])
        else:
            return tf.concat([previous, new[tf.newaxis, ...]], axis=0)

    def get_soft_max_exploration(self, state):
        action_nums = []
        for i in range(len(state)):
            if self.obs_mode == "hybrid":
                boolean_map, float_map, scalars = self.get_network_inputs_from_state(state[i])
                p = self.soft_explore_model(
                    [float_map[tf.newaxis, ...], boolean_map[tf.newaxis, ...], scalars[tf.newaxis, ...]]
                ).numpy()[0]
            else:
                boolean_map, float_map, _ = self.get_network_inputs_from_state(state[i])
                p = self.soft_explore_model([float_map[tf.newaxis, ...], boolean_map[tf.newaxis, ...]]).numpy()[0]
            action_num = np.random.choice(range(self.num_actions), size=1, p=p)
            action_nums.append(action_num)

        action_array = self.one_hot_action(action_nums)
        return action_array

    def act(self, state):
        return self.get_soft_max_exploration(state)

    def get_exploitation_action_target(self, state):
        a = None
        for i in range(len(state)):
            if self.obs_mode == "hybrid":
                boolean_map, float_map, scalars = self.get_network_inputs_from_state(state[i])
                new_a = self.exploit_model_target(
                    [float_map[tf.newaxis, ...], boolean_map[tf.newaxis, ...], scalars[tf.newaxis, ...]]
                ).numpy()[0]
            else:
                boolean_map, float_map, _ = self.get_network_inputs_from_state(state[i])
                new_a = self.exploit_model_target([float_map[tf.newaxis, ...], boolean_map[tf.newaxis, ...]]).numpy()[0]
            a = self._stack_along(a, new_a, i)
        return a

    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def soft_update(self, alpha):
        weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        self.target_network.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def get_network_inputs_from_state(self, state):
        if self.obs_mode == "image":
            boolean_map = tf.expand_dims(state[:, :, 0], axis=2)
            for i in range(1, self.boolean_map_shape[2] + 1):
                if i != 5:
                    boolean_map = tf.concat([boolean_map, tf.expand_dims(state[:, :, i], axis=2)], axis=2)
            float_map = tf.expand_dims(state[:, :, 5], axis=2)
            return boolean_map, float_map, None
        elif self.obs_mode == "hybrid":
            image_obs = state["image"]
            dense_obs = state["dense"]
            boolean_map = tf.expand_dims(image_obs[:, :, 0], axis=2)
            for i in range(2, self.boolean_map_shape[2] + 1):
                boolean_map = tf.concat([boolean_map, tf.expand_dims(image_obs[:, :, i], axis=2)], axis=2)
            float_map = tf.expand_dims(image_obs[:, :, 1], axis=2)
            return boolean_map, float_map, dense_obs
        else:
            return None, None, None

    def _extract_dicts_from_batch(self, batch, key1, key2):
        key1_batch, key2_batch = [], []
        for item in batch:
            key1_batch.append(item[key1])
            key2_batch.append(item[key2])
        return np.array(key1_batch), np.array(key2_batch)

    def get_network_inputs_from_batch_state(self, state):
        if self.obs_mode == "image":
            boolean_map = tf.expand_dims(state[:, :, :, 0], axis=3)
            for i in range(1, self.boolean_map_shape[2] + 1):
                if i != 5:
                    boolean_map = tf.concat([boolean_map, tf.expand_dims(state[:, :, :, i], axis=3)], axis=3)
            float_map = tf.expand_dims(state[:, :, :, 5], axis=3)
            return boolean_map, float_map, None
        elif self.obs_mode == "hybrid":
            image_obs, dense_obs = self._extract_dicts_from_batch(state, "image", "dense")
            boolean_map = tf.expand_dims(image_obs[:, :, :, 0], axis=3)
            for i in range(2, self.boolean_map_shape[2] + 1):
                boolean_map = tf.concat([boolean_map, tf.expand_dims(image_obs[:, :, :, i], axis=3)], axis=3)
            float_map = tf.expand_dims(image_obs[:, :, :, 1], axis=3)
            return boolean_map, float_map, dense_obs
        else:
            return None, None, None

    def train(self, experiences):
        state, action, reward, next_state, done = experiences

        if self.obs_mode == "image":
            boolean_map, float_map, _ = self.get_network_inputs_from_batch_state(state)
            next_boolean_map, next_float_map, _ = self.get_network_inputs_from_batch_state(next_state)
            q_star = self.q_star_model([next_float_map, next_boolean_map])

            with tf.GradientTape() as tape:
                q_loss = self.q_loss_model([float_map, boolean_map, action, reward, done, q_star])
        elif self.obs_mode == "hybrid":
            boolean_map, float_map, scalars = self.get_network_inputs_from_batch_state(state)
            next_boolean_map, next_float_map, next_scalars = self.get_network_inputs_from_batch_state(next_state)
            q_star = self.q_star_model([next_float_map, next_boolean_map, next_scalars])

            with tf.GradientTape() as tape:
                q_loss = self.q_loss_model([float_map, boolean_map, scalars, action, reward, done, q_star])
        else:
            print("\n\nERROR: Not implemented yet\n\n")

        q_grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))

        self.soft_update(self.params.alpha)

        return q_loss

    def save_weights(self, path_to_weights):
        self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.target_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.q_network.load_weights(path_to_weights)
        self.hard_update()

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()
