"""
Useful shared code functions
"""

from dataclasses import dataclass
from typing import Any, List
import cv2
from numpy import ndarray
from numpy import dtype, floating, integer, ndarray
from screeninfo import get_monitors

@dataclass
class Annotation:
    annotation_type : str
    center_x : int
    center_y : int
    width: int
    height: int

def get_optimal_font_scale(text : str, width : int) -> int:
    """
    Based on width of the current image, finds a good font size to display text as
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def get_optimal_window_scaling(width : int, height: int) -> float:
    """
    Based on the screen size, return the scaling that will fit the image to the screen, keeping the aspect ratio
    of the original window
    """
    # First, do we need to resize?
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    if width < screen_width and height < screen_height:
        # no resize needed!
        return 1
    width_scaling = screen_width/width
    height_scaling = screen_height/height
    return min(width_scaling,height_scaling)


def get_scaled_image(image : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]],
                      scale : float) -> cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]]:
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    scaled_image = cv2.resize(image, (width, height),interpolation=cv2.INTER_AREA)
    return scaled_image

def apply_infobar(frame : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]],
                        options : List[str],
                        frame_number : int,
                        width : int) -> None:
    """
    Applies the keyboard shortcuts, as well as the current frame number to the top left of the 
    current frame
    """
    text = ""
    for option in options:
        text = text + option + " | "

    # Add the frame number now
    str_frame : str = f'Frame : {frame_number}'
    text += str_frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = get_optimal_font_scale(text,width)
    color = (0, 0, 0)
    thickness = 2
    org = (10, 30) # (x, y) coordinates for bottom-left corner of the text
    return cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

def read_annotations(annotation_path : str) -> List[Annotation]:
    ret = []
    with open(annotation_path, "r") as file:
        for line in file:
            split_line = line.strip().split(" ")
            # convert center x center y width and height strings to ints
            ret.append(Annotation(split_line[0],
                       int(split_line[1]),
                       int(split_line[2]),
                       int(split_line[3]),
                       int(split_line[4])))
    return ret