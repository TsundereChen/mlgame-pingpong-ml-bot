"""
The template of the script for the machine learning process in game pingpong
"""
import tensorflow as tf
import numpy as np
import os

# Import the necessary modules and classes
from mlgame.communication import ml as comm

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
    curPath = os.path.abspath(os.path.dirname(__file__))
    model = tf.keras.models.load_model(curPath + '/saved_model/myModel')

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
            ballX = scene_info["ball"][0]
            ballY = scene_info["ball"][1]
            ball_speed_X = scene_info["ball_speed"][0]
            ball_speed_Y = scene_info["ball_speed"][1]
            platform_1P_X = scene_info["platform_1P"][0]
            blocker_X = scene_info["blocker"][0]
            features = [[ballX, ballY, ball_speed_X, ball_speed_Y, platform_1P_X, blocker_X]]
            prediction = model.predict(features)
            if prediction > 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                print("MOVE_RIGHT")
            elif prediction < 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
                print("MOVE_LEFT")
            else:
                comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_RIGHT"})
                print("NONE")

