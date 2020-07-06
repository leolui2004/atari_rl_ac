# import tensorflow as tf
# tf.device('/cpu:0') # force to run on cpu

import gym
import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Flatten, Conv2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
import random
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt

env = gym.make("Breakout-v4")
episode_limit = 5000
timestep_limit = 100000 # prevent endless game
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_step = episode_limit # reduce per episode
epsilon = 1.0
epsilon_reduce_step = (epsilon_start - epsilon_end) / epsilon_step
initial_replay = 200
verbose = 0
action_dim = env.action_space.n
action_space = [i for i in range(action_dim)] # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
discount = 0.97
actor_lr = 0.001
critic_lr = 0.001
random_step = 20
## input_dim = 3
frame_length = 4
frame_width = 84
frame_height = 84
score_list = [] # store scores for all episodes
step_list = [] # store timesteps for all episodes
score_avg_freq = 50
pretrain_use = 0
model_train = 1
model_save = 1
log_save = 1
actor_h5_path = 'atari_ac_actor.h5'
critic_h5_path = 'atari_ac_critic.h5'
actor_graph_path = 'atari_ac_actor.png'
critic_graph_path = 'atari_ac_critic.png'
policy_graph_path = 'atari_ac_policy.png'
log_path = 'atari_ac_log.txt'

def encode_initialize(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation_resize = np.uint8(resize(rgb2gray(processed_observation), (frame_width, frame_height)) * 255)
    state = [processed_observation_resize for _ in range(frame_length)]
    state_encode = np.stack(state, axis=0)
    return state_encode

def encode(observation, last_observation, state):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation_resize = np.uint8(resize(rgb2gray(processed_observation), (frame_width, frame_height)) * 255)
    state_next_return =  np.reshape(processed_observation_resize, (1, frame_width, frame_height))
    state_encode = np.append(state[1:, :, :], state_next_return, axis=0)
    return state_encode

input = Input(shape=(frame_length, frame_width, frame_height))
delta = Input(shape=[1])
con1 = Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu')(input) # 32 filter each 8x8 pixel, pooling with 4x4 and unchange padding size
con2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(con1) # filter number should double at each layer
fla1 = Flatten()(con2) # flatten to pass to dense
dense = Dense(128, activation='relu')(fla1) # shared by prob and value
prob = Dense(action_dim, activation='softmax')(dense) # actor and policy
value = Dense(1, activation='linear')(dense) # critic (q value)

def custom_loss(y_true, y_pred):
    out = K.clip(y_pred, 1e-8, 1-1e-8) # set boundary
    log_lik = y_true * K.log(out) # policy gradient
    return K.sum(-log_lik * delta)

if pretrain_use == 1:
    actor = load_model(actor_h5_path, custom_objects={'custom_loss': custom_loss}, compile=False)
    critic = load_model(critic_h5_path)

actor = Model(inputs=[input, delta], outputs=[prob]) # fit only, no predict
critic = Model(inputs=[input], outputs=[value]) # fit and predict
policy = Model(inputs=[input], outputs=[prob]) # probabiliy of action, predict only, no fit

actor.compile(optimizer=Adam(lr=actor_lr), loss=custom_loss)
critic.compile(optimizer=Adam(lr=critic_lr), loss='mean_squared_error')

def action_choose(state, epsilon, episode, action_space):
    if epsilon >= random.random() or episode < initial_replay: # set random play
        action = random.randrange(action_dim)
    else:
        probabiliy = policy.predict(state)[0]
        action = np.random.choice(action_space, p=probabiliy) # play by prob followed by policy predicted
    return action

def network_learn(state, action, reward, state_next, done):
    reward_clip = np.sign(reward) # clip reward to -1 and 1 for easier learning
    critic_value = critic.predict(state)
    critic_value_next = critic.predict(state_next)
    
    target = reward_clip + discount * critic_value_next * (1 - int(done)) # q value
    delta =  target - critic_value
    actions = np.zeros([1, action_dim])
    actions[np.arange(1), action] = 1
    
    actor.fit([state, delta], actions, verbose=verbose)
    critic.fit(state, target, verbose=verbose)

def epsilon_reduce(epsilon, episode):
    if epsilon > epsilon_end and episode >= initial_replay:
        epsilon -= epsilon_reduce_step
    return epsilon

def log(log_path, episode, timestep, score):
    logger = open(log_path, 'a')
    if episode == 0:
        logger.write('Episode Timestep Score\n')
    logger.write('{} {} {}\n'.format(episode + 1, timestep, score))
    logger.close()

for episode in range(episode_limit):
    observation = env.reset() # reset environment
    score = 0
    
    for _ in range(random.randint(1, random_step)):
        observation_last = observation
        observation, _, _, _ = env.step(0) # do nothing for random steps first to randomize starting environment
    state = encode_initialize(observation, observation_last)
    
    for timestep in range(timestep_limit):
        observation_last = observation
        action = action_choose(state[np.newaxis, :], epsilon, episode, action_space)
        observation, reward, done, _ = env.step(action)
        state_next = encode(observation, observation_last, state)
        if model_train == 1:
            network_learn(state[np.newaxis, :], action, reward, state_next[np.newaxis, :], done)
        state = state_next
        
        score += reward
        if done or timestep == timestep_limit - 1:
            score_list.append(score)
            step_list.append(timestep)
            if log_save == 1:
                log(log_path, episode, timestep, score)
            print('Episode {} Timestep {} Score {}'.format(episode + 1, timestep, score))
            break
    
    epsilon = epsilon_reduce(epsilon, episode)

if pretrain_use == 1:
    if model_save == 1:
        actor.save(actor_h5_path)
        critic.save(critic_h5_path)
else:
    if model_save == 1:
        actor.save(actor_h5_path)
        critic.save(critic_h5_path)
        plot_model(actor, show_shapes=True, to_file=actor_graph_path)
        plot_model(critic, show_shapes=True, to_file=critic_graph_path)
        plot_model(policy, show_shapes=True, to_file=policy_graph_path)

env.close()

xaxis = []
score_avg_list = []
step_avg_list = []
for i in range(1, episode_limit + 1):
    xaxis.append(i)
    if i < score_avg_freq:
        score_avg_list.append(np.mean(score_list[:]))
        step_avg_list.append(np.mean(step_list[:]))
    else:
        score_avg_list.append(np.mean(score_list[i - score_avg_freq:i]))
        step_avg_list.append(np.mean(step_list[i - score_avg_freq:i]))
        
plt.plot(xaxis, score_avg_list)
plt.show()

plt.plot(xaxis, step_avg_list)
plt.show()
