#Liberias
import tensorflow as tf
import numpy as np
import random
import gymnasium as gym
from collections import deque

class DQN:

    #------------------CONSTRUCTOR------------------

    #TODO: averigurar con que rellenarlo XD
    def __init__(self,num_states,num_actions,learning_rate=0.001, buffer_size=10000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.buffer = deque(maxlen=buffer_size)

        #Construccion de redes neuronales
        self.q_network = self.build_q_network(num_states,num_actions)
        self.q_target_network = self.build_q_network(num_states, num_actions)

        #Inicializar red objetvio
        self.q_target_network.set_wights(self.q_network.get_weights())

        #Optimizador
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
         


    #------------------RED NEURONAL------------------

    def build_q_network(self,num_states, num_actions):
        tf.random.set.seed(0)
        q_network = tf.keras.Sequential()
        q_network.add(tf.keras.layers.Input(shape=[num_states]))
        q_network.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        q_network.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        q_network.add(tf.keras.layers.Dense(num_actions, activation='linear'))
        return q_network


    #------------------E-GREEDY----------------------

    def greedy(self,state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            q_values= self.q_network(state)
            action = np.argmax(q_values)

        return action
    

    #------------Experiencia de repeticion-------------

    
    def actualizacionParametros(self,states, q_targets):
        with tf.GradientTape() as tape:
            q_values = self.q_network(states,training=True)
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q_values)
        gradients = tape.gradient(loss,self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.q_network.trainable_variables))   
        

    def experenciaRepeticion(self,buffer, batch_size, gamma):

        if len(buffer) <= batch_size:
            return
        
        minibatch = np.array(random.sample(buffer, batch_size), dtype='object')

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

        self.actualizacionParametros(states, q_values, self.q_network, self.optimizer)


    def add_experience(self, state, action, reward, next_state, done):
        
        self.buffer.append((np.array([state]), action, reward, np.array([next_state]), done))


    def update_target_network(self):
        
        self.q_target_network.set_weights(self.q_network.get_weights())

    
    

    #--------------------------Entrenamiento-----------------------------
    
    def entrenamiento(self,env,num_episodes=2000,batch_size=64,gamma=0.99,epsilon=1,epsilon_min=0.01,epsilon_decay=0.995,update_target_episode=100, print_every=50):

        epsilon = epsilon
        accumulate_reward = 0
        episode_rewards = []

        for episode in range(1, num_episodes + 1):
            state,_=env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:

                action = self.epsilon_greedy(np.array([state]), epsilon)

                next_state, reward , terminated,_, info = env.step(action)
                done = terminated

                steps += 1

                self.add_experience(state,action,reward,next_state,done)

                total_reward += 1

                self.experenciaRepeticion(batch_size, gamma)

                if steps % update_target_episode == 0:
                    self.update_target_network()

                state = next = state

            epsilon = max(epsilon_min,epsilon*epsilon_decay)

            accumulate_reward += total_reward
            average_reward = accumulate_reward
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



        

    