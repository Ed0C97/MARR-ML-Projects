import sys
from keras.models import load_model
import numpy as np
import time
import pygame

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)

pygame.display.set_mode((1600,1200))
def play(env, model):
    start_time = time.time()  # Inizia a registrare il tempo

    seed = 2000
    obs, _ = env.reset(seed=seed)

    # drop initial frames
    action0 = 0
    for i in range(50):
        obs, _, _, _, _ = env.step(action0)

    done = False
    total_reward = 0  # Inizializza la total reward
    while not done:
        obs_expanded = np.expand_dims(obs, axis=0)
        obs_expanded = obs_expanded / 255.0

        p = model.predict(obs_expanded)
        action = np.argmax(p)
        obs, reward, terminated, truncated, _ = env.step(action)  # Ottieni il reward ad ogni step

        total_reward += reward  # Aggiorna la total reward
        done = terminated or truncated

    end_time = time.time()  # Fine del tempo di esecuzione
    execution_time = end_time - start_time  # Calcola il tempo totale di esecuzione

    print(f"Tempo totale di esecuzione: {execution_time} secondi")
    print(f"Total reward: {total_reward}")

env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human'
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# your trained
# Sostituire con il percorso al tuo modello salvato
model_path = 'models_Oversampling_ON\model_cnn1_lr0.0001_batch64_epochs10.h5'
model = load_model(model_path)

play(env, model)



