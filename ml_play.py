"""
The template of the script for the machine learning process in game pingpong
"""
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
# Import the necessary modules and classes
from mlgame.communication import ml as comm

# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r)
  discounted_r = np.zeros_like(r)
  running_add = 0
  # we go from last reward to first one so we don't have to do exponentiations
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r) #normalizing the result
  discounted_r /= np.std(discounted_r) #idem
  return discounted_r

def ml_loop(side: str):
    """
    The main loop for the machine learning process

    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```

    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False

    """
    Create a new model
    """
    currentPath = os.path.dirname(os.path.abspath(__file__))
    modelFilename = "myModel"
    if os.path.exists(currentPath + "/" + modelFilename):
        print("LOAD MODEL....")
        model = tf.keras.models.load_model(currentPath + "/" + modelFilename)
    else:
        print("No model found, creating new model...")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=10, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(units=10, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(units=10, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    prevInput = None
    gamma = 0.99


    """
    Prepare variables needed
    """
    xTrain, yTrain, rewards = [], [], []
    rewardSum = 0.0
    episodeNum = 0
    resume = True
    runningReward = None
    epochsBeforeSaving = 10

    """
    Try to load previous model
    """

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()

    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()

        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False
            print('At run {}, the total reward was: {}'.format(episodeNum, rewardSum))
            episodeNum += 1
            model.fit(x = np.vstack(xTrain), y = np.vstack(yTrain), verbose = 1, sample_weight = discount_rewards(rewards, gamma))
            #model.fit(x = np.vstack(xTrain), y = np.vstack(yTrain), verbose = 1)

            if episodeNum % epochsBeforeSaving == 0:
                model.save(currentPath + "/" + modelFilename)

            xTrain, yTrain, rewards = [], [], []
            rewardSum = 0
            prevInput = None
            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information
        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_RIGHT"})
            ball_served = True
        else:
            currentInput = dataProcess(scene_info)
            x = currentInput - prevInput if prevInput is not None else np.zeros(10)
            prevInput = currentInput

            proba = model.predict(np.expand_dims(x, axis=1).T)
            action = "MOVE_LEFT" if np.random.uniform() < proba else "MOVE_RIGHT"
            y = 0 if action == "MOVE_LEFT" else 1
            xTrain.append(x)
            yTrain.append(y)

            """
            Send command
            """
            comm.send_to_game({"frame": scene_info["frame"], "command": action})
            rewardSum += 0.001
            rewards.append(rewardSum)


def dataProcess(scene_info):
    ballX       = scene_info["ball"][0]
    ballY       = scene_info["ball"][1]
    ballSpeedX  = scene_info["ball_speed"][0]
    ballSpeedY  = scene_info["ball_speed"][1]
    platform1PX = scene_info["platform_1P"][0]
    platform1PY = scene_info["platform_1P"][1]
    platform2PX = scene_info["platform_2P"][0]
    platform2PY = scene_info["platform_2P"][1]
    blockerX    = scene_info["blocker"][0]
    blockerY    = scene_info["blocker"][1]
    arr         = np.array([ballX, ballY, ballSpeedX, ballSpeedY,
        platform1PX, platform1PY, platform2PX, platform2PY, blockerX, blockerY])
    return arr
