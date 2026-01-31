# Game Reinforcement Learning

This project trains an AI agent to play Flappy Bird using Deep Q-Learning (DQN). The agent observes the game state (bird position, 
velocity, pipe locations) and learns optimal actions through trial and error. It receives rewards for passing pipes (+5), penalties for collisions (-10), 
and small rewards for survival (+0.1 per frame). Over thousands of episodes, the neural network learns patterns and strategies to maximize score, 
improving from random flapping to consistent gameplay.

## Setup

To get started, clone the repository and navigate to the project directory. Create a virtual environment with `python3 -m venv venv` and activate it using `source venv/bin/activate` 
(or `venv\Scripts\activate` on Windows). Install the required dependencies by running `pip install -r requirements.txt`, which includes pygame, numpy, and torch. 
Once setup is complete, train the agent with `python3 main.py --env flappy --mode train`, or watch a trained agent play using 
`python3 main.py --env flappy --mode test --model checkpoint.pth`. For faster training without the visual interface, add the flags `--render False --fps 1000`.
