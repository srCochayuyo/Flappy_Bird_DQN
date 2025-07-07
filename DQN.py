import tensorflow as tf
import numpy as np
import random
import gymnasium as gym
from collections import deque
import pickle


class DQN:

    #------------------CONSTRUCTOR------------------

    def __init__(self,num_states,num_actions,learning_rate=0.001, buffer_size=10000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.buffer = deque(maxlen=buffer_size)

        #Construccion de redes neuronales
        self.q_network = self.build_q_network(num_states,num_actions)
        self.q_target_network = self.build_q_network(num_states, num_actions)

        #Inicializar red objetvio
        self.q_target_network.set_weights(self.q_network.get_weights())

        #Optimizador
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Variables para guardar el mejor episodio
        self.best_episode = {
            'episode_number': 0,
            'total_reward': -float('inf'),
            'initial_state': None,  
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],  
            'dones': [],  
            'steps': 0
        }

    #------------------RED NEURONAL------------------

    def build_q_network(self,num_states, num_actions):
        q_network = tf.keras.Sequential()
        q_network.add(tf.keras.layers.Input(shape=[num_states]))
        q_network.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        q_network.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        q_network.add(tf.keras.layers.Dense(num_actions, activation='linear'))
        return q_network

    #------------------E-GREEDY----------------------

    def greedy(self,state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(self.num_actions)
            
        else:
            q_values= self.q_network(state)
            action = np.argmax(q_values,axis=1)[0]
            
        return action
    
    #------------Experiencia de repeticion-------------
    
    def actualizacionParametros(self,states, q_targets):
        with tf.GradientTape() as tape:
            q_values = self.q_network(states,training=True)
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q_values)
        gradients = tape.gradient(loss,self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.q_network.trainable_variables))   
        
    def experenciaRepeticion(self, batch_size, gamma):
        if len(self.buffer) <= batch_size:
            return
        
        minibatch = np.array(random.sample(self.buffer, batch_size), dtype='object')
        states = np.vstack(minibatch[:,0])
        actions = minibatch[:,1].astype(int)
        rewards = minibatch[:,2]
        next_states = np.vstack(minibatch[:,3])
        dones = minibatch[:,4]

        q_values = self.q_network(states).numpy()
        next_q_values = self.q_target_network(next_states).numpy()
        q_targets = rewards + (gamma * np.max(next_q_values, axis=1) * (1-dones))
        q_values[np.arange(batch_size), actions] = q_targets
        q_values = tf.convert_to_tensor(q_values)

        self.actualizacionParametros(states, q_values)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update_target_network(self):
        self.q_target_network.set_weights(self.q_network.get_weights())

    #--------------------------Entrenamiento-----------------------------
    
    def entrenamiento(self,env,num_episodes=3000,batch_size=32,gamma=0.99,epsilon=1,epsilon_min=0.01,epsilon_decay=0.995,update_target_episode=10, print_every=50):

        epsilon = epsilon
        accumulate_reward = 0
        episode_rewards = []
        maxTotalReward = 5

        for episode in range(1, num_episodes + 1):
            state,_=env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            
            current_initial_state = state.copy()  
            current_states = []
            current_actions = []
            current_rewards = []
            current_next_states = [] 
            current_dones = []  

            while not done:
                
                current_states.append(state.copy())
                
                action = self.greedy(np.array([state]), epsilon)
                
             
                current_actions.append(action)

                next_state, reward , terminated,_, info = env.step(action)
                done = terminated

                
                current_rewards.append(reward)
                current_next_states.append(next_state.copy())
                current_dones.append(done)

                steps += 1

                self.add_experience(state,action,reward,next_state,done)

                total_reward += reward

                self.experenciaRepeticion(batch_size, gamma)

                if episode % update_target_episode == 0:
                    self.update_target_network()

                state = next_state

            
            if total_reward > maxTotalReward:
                maxTotalReward = total_reward
                
                
                self.best_episode = {
                    'episode_number': episode,
                    'total_reward': total_reward,
                    'initial_state': current_initial_state,
                    'states': current_states,
                    'actions': current_actions,
                    'rewards': current_rewards,
                    'next_states': current_next_states,
                    'dones': current_dones,
                    'steps': steps
                }
                
                
                self.save_model('PIPO.keras')
                self.save_best_episode('mejor_episodio.pkl')
                
                print(f"-----GUARDADO DE MEJOR MODELO DQN | Nueva Maxima recompensa: {maxTotalReward} | Episodio {episode}")

            epsilon = max(epsilon_min,epsilon*epsilon_decay)

            accumulate_reward += total_reward
            average_reward = accumulate_reward / episode
            episode_rewards.append((episode,total_reward))

            if episode % print_every == 0:
                print(f"Episode: {episode} | Total reward: {total_reward:.3f} | "
                      f"Average reward: {average_reward:.3f} | Steps: {steps} | "
                      f"Epsilon: {epsilon:.3f}")
                
        env.close()

        return episode_rewards
        
    #---------------------Guardado y carga del modelo-------------------------------

    def save_model(self, filepath):
        self.q_network.save(filepath)
    
    def load_model(self, filepath):
        self.q_network = tf.keras.models.load_model(filepath)
        self.q_target_network = self.build_q_network(self.num_states, self.num_actions)
        self.q_target_network.set_weights(self.q_network.get_weights())
        return self
    
    #---------------------Guardado y carga del mejor episodio-----------------------
    
    def save_best_episode(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_episode, f)
        print(f"Mejor episodio guardado en {filepath}")
    
    def load_best_episode(self, filepath):
        with open(filepath, 'rb') as f:
            self.best_episode = pickle.load(f)
        print(f"Mejor episodio cargado desde {filepath}")
        return self.best_episode
    
    def replay_best_episode(self, env, render=True, delay=0.1):
        
        
        if not self.best_episode['states']:
            print("No hay episodio guardado para reproducir")
            return
        
        print(f"Reproduciendo mejor episodio #{self.best_episode['episode_number']}")
        print(f"Recompensa total: {self.best_episode['total_reward']}")
        print(f"Pasos: {self.best_episode['steps']}")
        print("-" * 50)
        
        # Reiniciar el entorno
        env.reset()
        
        # Intentar establecer el estado inicial si es posible
        if hasattr(env, 'set_state') and self.best_episode['initial_state'] is not None:
            env.set_state(self.best_episode['initial_state'])
        
        total_reward = 0
        
        for step, (state, action, reward) in enumerate(zip(
            self.best_episode['states'], 
            self.best_episode['actions'], 
            self.best_episode['rewards']
        )):
            if render:
                env.render()
                
            
            print(f"Step {step + 1}: Action = {action}, Reward = {reward}")
            
            # Ejecutar la acción
            next_state, actual_reward, terminated, truncated, info = env.step(action)
            total_reward += actual_reward
            
            if terminated or truncated:
                break
        
        print(f"\nReproducción completada. Recompensa total: {total_reward}")
        if render:
            env.render()
            
        
        return total_reward
    
    def replay_best_episode_visual_only(self, env, delay=0.1):
       
        import time
        
        if not self.best_episode['actions']:
            print("No hay episodio guardado para reproducir")
            return
        
        print(f"Reproduciendo mejor episodio #{self.best_episode['episode_number']}")
        print(f"Recompensa esperada: {self.best_episode['total_reward']}")
        print(f"Pasos: {self.best_episode['steps']}")
        print("-" * 50)
        
       
        best_match_reward = -float('inf')
        best_seed = None
        
      
        for seed in range(10):
            env.reset(seed=seed)
            test_reward = 0
            test_steps = 0
            
            for action in self.best_episode['actions']:
                next_state, reward, terminated, truncated, info = env.step(action)
                test_reward += reward
                test_steps += 1
                
                if terminated or truncated:
                    break
            
            if test_reward > best_match_reward:
                best_match_reward = test_reward
                best_seed = seed
        
     
        print(f"Usando semilla {best_seed} (recompensa: {best_match_reward:.3f})")
        
        env.reset(seed=best_seed)
        total_reward = 0
        
        for step, action in enumerate(self.best_episode['actions']):
            env.render()
            
            
            print(f"Step {step + 1}: Action = {action}")
            
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        env.render()
  
        
        print(f"\nReproducción completada.")
        print(f"Recompensa original: {self.best_episode['total_reward']}")
        print(f"Recompensa obtenida: {total_reward}")
        
        return total_reward
    
    def get_best_episode_summary(self):
        if not self.best_episode['states']:
            return "No hay episodio guardado"
        
        summary = {
            'episode_number': self.best_episode['episode_number'],
            'total_reward': self.best_episode['total_reward'],
            'steps': self.best_episode['steps'],
            'average_reward_per_step': self.best_episode['total_reward'] / self.best_episode['steps'],
            'actions_taken': self.best_episode['actions'],
            'rewards_per_step': self.best_episode['rewards']
        }
        
        return summary
    
    