import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
import gym
from gym import spaces
import time
import gc
import os

# Encoded data "encoded_training_data" each step consists of 20 data points
# Each time step had the original 2D size of(1440,6). Using an encoder model, they "attributes" are extracted into a (1,20) array (for each time step)
egitim_verileri = np.load(os.path.expanduser('~')+'/encoded_training_data.npy')  
egitim_verileri = np.expand_dims(egitim_verileri, axis=1)

# This is the market data. Consists of "high,open,low,close values of bitcoin market data"
etiket_verileri = np.load(os.path.expanduser('~')+'/egitim_etiket.npy') 
etiket_verileri = etiket_verileri[:len(egitim_verileri)]

# Check the GPUs
try:
    tf.config.set_visible_devices([], 'GPU')
    logical_devices = tf.config.get_visible_devices()
    print("Visible devices:", logical_devices)
except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

#%% Simulation
def simulasyon(action,baslangic_idx):
    # nda
    # Placeholder simulation function.
    # A reward is calculated; using the pair, commision, no postion penalty stop point etc.
    
    parite = "BTCUSDT" # The pair we'll be trading
    commision = 0.0009 # Commision for market buy - sell orders

    aksiyon_sayisi = 3 # 3 Actions: Long, Short, No position
    nopos_penaltisi = -12 # penalty for not taking an action
    
    tp_degerleri = 3 # Take profit at %3
    stop_degeri = 1.5 # Stop loss at %1.5
    leverage = 10   # leverage
    position_amt = 100 # position amount in dollars
    
    reward = 0
    step = 0
    return reward, step # eğitim modunda return. reward ve siradaki adimin indisi

# Initialize to get the first sequence from the dataset
def initialize_state():
    global current_index  # Track the current step in the dataset
    current_index = 0  # Reset to the beginning of the dataset
    state = egitim_verileri[current_index]
    return state

# This function moves to the next step and fetches the next sequence of data.
def get_next_state():
    global current_index
    if current_index > len(egitim_verileri): current_index = 0
    state = egitim_verileri[current_index]
    return state

# This function determines when an episode ends.
# Episode ends when we reach to end of the dataset
def check_if_done():
    global current_index
    if current_index >= len(egitim_verileri):  # End if there’s not enough data left
        return True
    return False

#%% RL
# Define the Policy Network
# A mix of 1 Conv1D layer and a single LSTM layer, with 2 dense layers gave the best results,
# while keeping the training fast. More models like Siamese networks and Transformers, or more complex Conv and LSTM models gave worse results
# Since our data is stock market data and has a lot of "noise", complex algorithms prone to overfit on this dataset
def create_policy_network(sequence_length, input_size, num_actions):
    ####################################################################################################################
    inputs = Input(shape=(sequence_length, input_size))    
    x = Conv1D(filters=512, kernel_size=3, activation='tanh', padding='same')(inputs)    
    x = LSTM(512, return_sequences=False)(x)    
    dense = Dense(64, activation='tanh')(x)    
    action_probs = Dense(num_actions, activation='softmax', name="action_probs")(dense)
    value = Dense(1, name="value")(dense)
    ####################################################################################################################

    return Model(inputs, [action_probs, value])

# Custom Environment
# It is used to set action size etc.
# We also run our simulation in the step function and adjust the next step, reward, etc. accordingly
class StockTradingEnv(gym.Env):
    def __init__(self):
        super(StockTradingEnv, self).__init__()
        self.sequence_length = 1
        self.input_size = 20
        self.num_actions = 3  # Long, Short, Nopos
        
        # Define action and observation spaces
        # We have discrete actions consisting of 3 actions.
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sequence_length, self.input_size), dtype=np.float32
        )
        
        # Internal state
        self.state = None
        self.done = False

    def step(self, action):
        global current_index
        reward,current_index = simulasyon(action,current_index)  # Simulation logic. Should return the reward and the next index
        self.done = check_if_done()
        self.state = get_next_state()         
        return self.state, reward, self.done, {}

    def reset(self):
        # Reset the environment to start a new episode
        self.state = initialize_state()
        self.done = False # set internal state
        return self.state


# PPO requires maintaining two networks: a policy network and an old policy network for comparison. 
# The policy network is trained to maximize the advantage function.
# I guess this is trying to produce a better network by comparing it to the old network.
class PPOAgent:
    def __init__(self, sequence_length, input_size, num_actions, learning_rate=0.0003, gamma=0.99, clip_ratio=0.3):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
        # Create policy and old policy networks
        self.policy_network = create_policy_network(sequence_length, input_size, num_actions)
        self.old_policy_network = create_policy_network(sequence_length, input_size, num_actions)
        self.old_policy_network.set_weights(self.policy_network.get_weights()) # Use the old weights
        
        # Adam Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Basically, calculates how much better, or worse, a action is, compared to value function prediction
    # High gamma value means future rewards are less valuable than immediate ones
    def compute_advantages(self, rewards, values, dones):
        advantages = []
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                discounted_sum = 0
            delta = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(rewards) else 0) - values[t] # Temporal difference (TD) error. Measures how good the current state-action pair is
            discounted_sum = delta + self.gamma * discounted_sum
            advantages.insert(0, discounted_sum)
        return np.array(advantages)
    
    
    def train(self, states, actions, rewards, values, dones):
        # Compute advantages
        advantages = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert data to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        values = tf.convert_to_tensor(values, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # Compute old action probabilities outside the gradient tape
        old_action_probs, _ = self.old_policy_network(states)
        old_action_probs = tf.stop_gradient(old_action_probs)
        
        with tf.GradientTape() as tape:
            # Forward pass through current policy network
            action_probs, values_pred = self.policy_network(states)
            
            # Compute log probabilities
            actions_one_hot = tf.one_hot(actions, depth=action_probs.shape[-1])
            selected_probs = tf.reduce_sum(action_probs * actions_one_hot, axis=-1)
            old_selected_probs = tf.reduce_sum(old_action_probs * actions_one_hot, axis=-1)
            
            # PPO ratio
            ratios = tf.exp(tf.math.log(selected_probs + 1e-10) - tf.math.log(old_selected_probs + 1e-10))
            
            # Surrogate loss
            surrogate1 = ratios * advantages
            surrogate2 = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Value loss
            value_loss = tf.reduce_mean(tf.square(rewards - tf.squeeze(values_pred)))
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
        
        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))
        
        # Update old policy network
        self.old_policy_network.set_weights(self.policy_network.get_weights())


#%% Train
env = StockTradingEnv()
agent = PPOAgent(sequence_length=1, input_size=20, num_actions=3, learning_rate=0.0003, gamma=0.90, clip_ratio=0.75)

num_episodes = 1000000
batch_size = 128
best_reward = 0
global current_index
current_index = 0
total_total_reward = []

# We will check the moving average of rewards to see ih the rewards are getting better or worse.
def moving_average(total_rewards, window=100):
    return np.array([np.mean(total_rewards[max(0, i-window+1):i+1]) for i in range(len(total_rewards))])

for episode in range(num_episodes):
    sayac = 0
    start = time.time()
    state = env.reset()
    states, actions, rewards, values, dones = [], [], [], [], []
    total_reward = 0  # Track total reward for this episode
    son_odul = 0
    done = False
    sayac_suresi = time.time()
    while not done:        
        sayac += 1
        
        # Predict action probabilities and value
        # The current state, (the stock market status) which is our input data, is given to the network. An action will be selected according to the model we have trained.
        # The result of the action probe is something like [0.30,0.40,0.30]. But it doesn't necessarily mean that the value "0.40" will be selected. I explained it below.
        state_input = np.expand_dims(state, axis=0)  # Add batch dimension
        
        # !!!!!!!! Using the commented line below causes a memory leak.
        # action_probs, value = agent.policy_network.predict(state_input,verbose = False)
        action_probs, value = agent.policy_network(state_input, training=False)
        

        # Action is not selected deterministically "It is sampled from the probability distribution (hence, the term "sample")."
        # The decision is made using the value returnd from the policy network. BUT there is no obligation to select the action with highest probability.
        # The action with the maximum value has the highest probability of being selected. But it does not have to be selected. The others also have a probability
        # It allows for more "exploration", instead of more "exploitation".
        action_probs = action_probs.numpy()
        action = np.random.choice(action_probs.shape[-1], p=action_probs[0])
        
        
        # calculate the reward, set the next step
        next_state, reward, done, _ = env.step(action)

        # Record data and next state
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value[0][0])
        dones.append(done)
        
        state = next_state
        
        # just to print the reward of each episode on the screen. it has no effect on training
        total_reward += reward 
        son_odul += reward

    # Train PPO with collected batch
    agent.train(np.array(states), np.array(actions), np.array(rewards), np.array(values), np.array(dones))
    
    # Delete per-episode to free memory
    del states, actions, rewards, values, dones, action_probs, action
    gc.collect()
    tf.keras.backend.clear_session()
    
    # These have no effect on education either. Just to see the results
    total_total_reward.append(total_reward)
    smoothed_rewards = moving_average(total_total_reward, window=200)
    if len(smoothed_rewards) > 0:temp_sr = smoothed_rewards[-1]
    else :temp_sr = 0
    
    # will save the weights that gave the best reward
    if total_reward > best_reward:
        best_reward = total_reward
        agent.policy_network.save("path.h5")
        
    end = time.time()
    sure = end - start
    
    print(f"Episode {episode + 1} finished in {sure:.1f}s. Total Reward: {total_reward:.0f}. Average(W=200): {temp_sr:.1f}")

