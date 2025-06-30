#Librerias
import gymnasium as gym
import flappy_bird_gymnasium
from DQN import DQN



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



##----------------------------ENTORNO------------------------------------------

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)


num_states = env.observation_space.shape[0]
num_actions = env.action_space.n


DQN_Agent = DQN(num_states=num_states, num_actions=num_actions)

#Entrenamiento
train = DQN_Agent.entrenamiento(env=env,num_episodes=2000,epsilon_decay=0.9995,print_every=50)

DQN_Agent.save_model('flappybird_qNetwork.keras')

#-----------------------------------------------------------------------------