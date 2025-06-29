#Librerias
import gymnasium as gym
import flappy_bird_gymnasium


"""
|ACCIONES|
|
|0 --> No hace nada|
|1 --> Salta/Vuela |

---------------------------------------------------------------

|RECOMEPENSAS|
|               
| +0.1 --> Cada frame que se mantiene vivo|
| +1.0 --> Cada tuberia pasada
| -1.0 --> Cada vez que muere  
| -0.5 --> Cada vez que toca la parte superior de la pantalla

"""



env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

state, _ = env.reset()
total_reward = 0
steps = 0
episode = 0

while True:

    action = env.action_space.sample()

    state, reward, terminated, _, info = env.step(action)

    total_reward += reward
    episode += 1
    steps += 1

    print("Episode:",episode," Total Reward:",total_reward," Steps:",steps)
    

    if terminated:
        break


env.close()