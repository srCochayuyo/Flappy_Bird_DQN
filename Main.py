#Librerias
import gymnasium as gym
import flappy_bird_gymnasium
from DQN import DQN




"""
|ACCIONES|
|
|0 --> No hace nada
|1 --> Salta/Vuela 

---------------------------------------------------------------

|RECOMEPENSAS|
|               
| +0.1 --> Cada frame que se mantiene vivo
| +1.0 --> Cada tuberia pasada
| -1.0 --> Cada vez que muere  
| -0.5 --> Cada vez que toca la parte superior de la pantalla

"""


#---------------------------ENTORNO------------------------------------------

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)



#----------------------------DQN------------------------------------------------

num_statesDQN = env.observation_space.shape[0]
num_actionsDQN = env.action_space.n

DQN_Agent = DQN(num_states=num_statesDQN, num_actions=num_actionsDQN)



#train = DQN_Agent.entrenamiento(env=env,num_episodes=10000,epsilon_decay=0.995,print_every=50)

                                                                                                              

#-----------------------------------------CARGAR MEJOR MODELO---------------------------------------------------------------


print("REPRODUCIENDO EL MEJOR EPISODIO")
env_replay = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
DQN_Agent.get_best_episode_summary()
DQN_Agent.load_best_episode('mejor_episodio.pkl')
DQN_Agent.replay_best_episode_visual_only(env_replay, delay=0.05)

env_replay.close()

