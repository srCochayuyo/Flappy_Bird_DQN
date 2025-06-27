#Librerias
import gymnasium as gym
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

obs, _ = env.reset()
while True:

    action = env.action_space.sample()

    obs, reward, terminated, _, info = env.step(action)
    

    if terminated:
        break

env.close()