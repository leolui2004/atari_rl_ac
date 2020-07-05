# Actor-Critic Reinforcement Learning for Breakout-v4 (Atari)

My fifth project on github. Reinforcement Learning with Actor-Critic to play Breakout-v4 (Atari) from OpenAI Gym.

Feel free to provide comments, I just started learning Python for 2 monnths and I am now concentrating on data anylysis and web presentation.

## Result
Result
I just played 5000 games because it is too long (over 8 hours), but there is already a trend of improvement, although it is not so obvious.

![image](https://github.com/leolui2004/atari_rl_ac/blob/master/atari_ac_v1_score.png)

With x-axis being the episode (number of plays) and y-axis is the score, the line is the average score of the last 50 episodes. The result shows a little improvement on average scores from around 1.2 to 2.0 after 5000 episodes.

However for the graph of Timestep also in average of 50 episodes, the result is brilliant.

![image](https://github.com/leolui2004/atari_rl_ac/blob/master/atari_ac_v1_timestep.png)

## Methodology
### Gameplay Part
1. Reset environment on each episode start (game, scores, etc.)
2. Do nothing for a random steps (I just set a random between 1-20 here) to randomize the starting environment
3. Predict an action from the model and act on it
4. Observation, reward received after an action are passed to model for training
5. Record the reward and go back to step 1 again until an episode ends (all life lost)

![image](https://github.com/leolui2004/atari_rl_ac/blob/master/atari_ac_v1_actor.png)

### Reinforcement Learning Part
1. Images are resized to 84x84, combining 4 consecutive frames as one input so as to represent an action not just a particular state for easier learning
2. A2C model is being used, starting with 2 Convolutional Layers, 1 Flatten Layer, 1 Dense Layer, parameters are shown on graph above
3. Actor, Policy and Critic share same model built on step 2
3a. Actor and Policy use a softmax layer at the end (connect to Dense Layer) to output an array of probability of actions
3b. Critic use a linear activation (connect to Dense Layer) and only output a floating value
4. Custom Loss for Actor Model is defined following Policy Gradient Algorithm
5. Epsilon-greedy is used to provide linear randomness to the action, starting with a chance of 1.0 to 0.1 at the end, also for the first 200 episodes the actions are fully random so as to provide enough data for the model to start learning
