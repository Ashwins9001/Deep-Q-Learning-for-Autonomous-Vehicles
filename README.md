# Deep-Q-Learning-on-Carla
Testing a double deep q-network for self-driving cars using open-source Carla environment. Code mainly generated from open-source authors (Felippe Roza) with slight modifications (based on Simonini Thomas's deep q-learning implementation in Space Invaders) to memory buffer (no priority replay as Felippe Roza did) and network architecture (no dueling added in, thus no advantage or value functions). Used as a learning experience to apply knowledge about reinforcement learning. 

<p align="center">
  <img src="https://github.com/Ashwins9001/Deep-Q-Learning-on-Carla/blob/master/Pictures/Render.JPG" width="420"/> 

</p>

Quite a bit is boilerplate code to allow Tensorflow to interface with the Carla environment. Relevant functions that implement deep learning include:

- [x] class DQNetwork() -> Implement architecture of Deep Q-Network. Contains convolution layers, followed with ELU activation (continuous q-val prediction), and fully-connected layers. Final output fit to action-space to ensure every possible action has associated q-val. 
- [x] class Memory() -> Create memory buffer to implement Experience Replay and train agent with greater variance
- [x] def update_target_graph() -> Use fixed weights for training to ensure weight convergence, update every 5000 steps 
- [x] def testing(map, vehicle, sensors) -> Test agent after training 
- [x] def training(map, vehicle, sensors) -> Implement evaluation & target net, select actions from evaluation & determine q-value using target. Design creates double DQN which reduces reward overestimation to help agent ignore less important states. Generate experiences from memory buffer, compute target q-values using Bellman equation, compute loss and train evaluation network. The agent is initially set on self-driving mode to fill the memory buffer, then it begins training the evaluation network. Upon colliding with any object, the agent's environment gets reset and it spawns randomly on the map. 
- [x] def process_image(queue) -> Convert input image from Carla into numpy array for neural net processing   
- [x] class Sensors(object) -> Use lambda func to listen for collision, lane crossing, and RGB images by anchoring sensor objects to actor (car)

## Results
After training for 50 episodes with 100 steps per episode and a memory buffer size of 1000, results showed no obvious trend. 

A large source of error is the constantly changing environment. The car respawns randomly upon every episode to ensure the agent is exposed to many new experiences, however due to this the rewards and subsequent loss will also highly vary. The only solution is to attempt training on many more episodes with a larger memory buffer to look for trends. 

<p align="center">
  <img src="https://github.com/Ashwins9001/Deep-Q-Learning-on-Carla/blob/master/Pictures/Training_Loss.png" width="420"/> 
  <img src="https://github.com/Ashwins9001/Deep-Q-Learning-on-Carla/blob/master/Pictures/Training_Reward.png" width="420"/>
</p>

## Acknowledgments
Project used for educational purposes to understand, implement and experiment with deep q-networks. Code is combination from other authors' work who are accredited below.

Thank you for keeping your contributions open-source for others to learn from.

Implementation heavily based on Felippe Roza's Double Duel DQN with Priority Experience Replay: https://github.com/FelippeRoza/carla-rl

Neural network architecture (DQNetwork() class), predict_action(), map_from_control(), testing(), training(), process_image(), update_target_graph(), compute_reward(), render_loop(), map_action(), map_from_control(); Memory() class's fill_memory(), save_memory(), load_memory() functions attributed to Felippe's work.

Referenced Simonini Thomas's Deep Learning course for learning to interface Tensorflow with Gym environments: https://github.com/simoninithomas/Deep_reinforcement_learning_Course

Memory() class's init(), add(), sample(); predict_action() functions attributed to Thomas's work.

Referenced Arthur Juliani for update_target_graph() function, more found from Deep Reinforcement Learning Tutorials page: https://github.com/awjuliani/oreilly-rl-tutorial

Additionally referenced Memory() class's init(), add(), sample(); predict_action() (epsilon-greedy implementation) functions & code structure from Udacity's Deep Learning Course: https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb

Remaining functions come from Carla documentation to interface with server.
