import logging
import numpy as np

from rlearn.sports.soccer.constant import STOP_THRESHOLD
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def discretize_direction(velocity_x: float, velocity_y: float) -> str:
    """
    Discretize the direction of the ball/player into 8 directions
        - idle: 0 (when velocity is below STOP_THRESHOLD)
        - right: 1
        - up_right 2
        - up: 3
        - up_left: 4
        - left: 5
        - down_left: 6
        - down: 7
        - down_right: 8
    """

    # if velocity is below threshold, then idle
    if np.sqrt(velocity_x**2 + velocity_y**2) < STOP_THRESHOLD:
        return "idle"

    # calculate angle
    angle = np.arctan2(velocity_y, velocity_x)
    angle = np.rad2deg(angle)
    angle = (angle + 360) % 360

    # discretize angle into 8 directions
    if 22.5 <= angle < 67.5:
        direction = "up_right"
    elif 67.5 <= angle < 112.5:
        direction = "up"
    elif 112.5 <= angle < 157.5:
        direction = "up_left"
    elif 157.5 <= angle < 202.5:
        direction = "left"
    elif 202.5 <= angle < 247.5:
        direction = "down_left"
    elif 247.5 <= angle < 292.5:
        direction = "down"
    elif 292.5 <= angle < 337.5:
        direction = "down_right"
    else:
        direction = "right"

    return direction