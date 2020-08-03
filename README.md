# Deep-Q-Learning-on-Carla
Testing a double deep q-network for self-driving cars using open-source Carla environment. Code mainly generated from open-source authors with some modification to simplify it. Used as a learning experience to apply knowledge about reinforcement learning. 

Quite a bit is boilerplate code to allow Tensorflow to interface with the Carla environment. Relevant functions that implement deep learning include:

- [x] class DQNetwork() -> Implement architecture of Deep Q-Network. Contains convolution layers, followed with ELU activation (continuous q-val prediction), and fully-connected layers. Final output fit to action-space to ensure every possible action has associated q-val. 
- [x] class Memory() -> Create memory buffer to implement Experience Replay and train agent with greater variance
- [x] def update_target_graph() -> Use fixed weights for training to ensure weight convergence, update every 5000 steps 
- [x] def testing(map, vehicle, sensors) -> Test agent after training 
- [x] def training(map, vehicle, sensors) -> Implement evaluation & target net, select actions from evaluation & determine q-value using target. Design creates double DQN which reduces reward overestimation to help agent ignore less important states. Generate experiences from memory buffer, compute target q-values using Bellman equation, compute loss and train evaluation network. 
- [x] def process_image(queue) -> Convert input image from Carla into numpy array for neural net processing   
- [x] class Sensors(object) -> Use lambda func to listen for collision, lane crossing, and RGB images by anchoring sensor objects to actor (car)

## Acknowledgments
Implementation heavily based on Felippe Roza's Double DQN with Priority Experience Replay: https://github.com/FelippeRoza/carla-rl
Referenced Simonini Thomas's Deep Learning course for learning to interface Tensorflow with Gym environments: https://github.com/simoninithomas/Deep_reinforcement_learning_Course
