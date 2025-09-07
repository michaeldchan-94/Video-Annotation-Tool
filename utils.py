"""
Shared code
"""


from dataclasses import dataclass
from typing import Any, List
import cv2
from numpy import ndarray
from numpy import dtype, floating, integer, ndarray

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
        color = (255, 255, 255) # White color (B, G, R)
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