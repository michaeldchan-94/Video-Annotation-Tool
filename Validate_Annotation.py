"""
This is a tool that can be used to replay the annotations produced by VideoAnnotator
Takes in a video file path and it's corresponding annotation file path

If annotation path is not passed in, defaults to the same directory as the passed in video file
"""


import os
import sys
from typing import Any, List
from numpy import dtype, floating, integer, ndarray
import cv2
from utils import apply_infobar, get_optimal_font_scale, read_annotations,Annotation
from pynput.keyboard import Key, Controller, Listener

class AnnotationValidator:
    def __init__(self):
        self.annotations : List[Annotation]
        self.frame_copy : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self.cur_frame : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self.normal_keyboard_options : List[str] = ["n : Next Frame","N : Forward 10 Frames","p : Previous Frame", "P : Back 10 Frames"]
        self.get_next_frame : bool = True
        self.frame_number : int = 0
        self.caps_or_shift_active : bool = False

    def ReadAnnotations(self, video_path : str, annotation_path : str = "") -> None:
        full_video_name : str = os.path.basename(video_path)
        video_name : str = os.path.splitext(full_video_name)[0]

        # There's a known bug in opencv where shift + keys result in the lower value
        # in order to work around this, I had to create a separate listener to handle the 
        # shift + keystroke cases
        listener : Listener = Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        if annotation_path != "":
            # if an annotation path is provided, use that path
            self.annotations = read_annotations(annotation_path)
        else:
            # otherwise, it is in same location as video
            directory_path = os.path.dirname(video_path)
            annotation_name = video_name + ".annotations"
            annotation_path = os.path.join(directory_path,annotation_name)
            self.annotations = read_annotations(annotation_path)

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            # Only get the next frame if appropriate key bindings have been pressed
            if (self.get_next_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
                ret, frame = cap.read()   
                if not ret:
                    print("End of video.")
                    break

                # Make a clean copy in case we need to go back to clean image
                self.frame_copy = frame.copy()
                if self.frame_number == 0:
                    self.width = int(frame.shape[1])
                    self.height = int(frame.shape[0])
                    
                self.cur_frame = self.frame_copy.copy()
                self.__apply_annotation(self.cur_frame, self.annotations[self.frame_number])
                
                apply_infobar(self.cur_frame ,self.normal_keyboard_options,self.frame_number,self.width)

            cv2.imshow('Video Stream', self.cur_frame)

            self.get_next_frame = False
            key = cv2.waitKey(0) & 0xFF
            character = chr(key)
            # First, I have to check if the character is a letter and if Shift/Caps Lock is active
            if character.isalpha() and self.caps_or_shift_active:
                if character.upper() == "N":
                    self.frame_number += 10
                    self.get_next_frame = True
                    continue
                elif character.upper() == "P":
                    self.frame_number = max(0, self.frame_number - 10)
                    self.get_next_frame = True
                    continue
            # Otherwise, handle regular cases
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.frame_number += 1
                self.get_next_frame = True
            elif key == ord('p'):
                self.frame_number = max(0, self.frame_number - 1)
                self.get_next_frame = True

    def __apply_annotation(self, frame : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]],
                           annotation : Annotation):
        """
        Given an annotation, applies it to the given frame
        If it object is visible i.e ("V") draw the bbox that is there
        otherwise, display the status of the missing annotation
        S : Skipped
        I : Invisible
        as well as the corresponding frame
        """
        if (annotation.annotation_type == "V"):
            bottom_x : int = annotation.center_x-int(annotation.width/2)
            bottom_y : int = annotation.center_y-int(annotation.height/2)
            upper_x : int = annotation.center_x+int(annotation.width/2)
            upper_y : int = annotation.center_y+int(annotation.height/2)
            frame = cv2.rectangle(frame, (bottom_x,bottom_y),(upper_x,upper_y), (0, 255, 0), 2)
        else:
            text : str = f'Annotation is {annotation.annotation_type} for Frame : {self.frame_number}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = get_optimal_font_scale(text,self.width)
            color = (0, 0, 255) # White color (B, G, R)
            thickness = 2
            org = (0, int(self.height * 0.95)) # (x, y) coordinates for bottom-left corner of the text
            return cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    def onPress(self,key):
        """
        Just a listener that also catches keystrokes, but this one just toggles if shift has been hit
        """
        if key == Key.shift_l or key == Key.shift_r:
            self.caps_or_shift_active = True
    
    def onRelease(self,key):
        """
        Another event listener for key releases, just checks if shift key was just released
        as well as if caps was hit
        """
        if key == Key.shift_l or key == Key.shift_r:
            self.caps_or_shift_active = False
        elif key == Key.caps_lock:
            keyboard_controller = Controller()
            self.caps_or_shift_active = keyboard_controller.caps_lock


if __name__ == "__main__":
    annotation_validator = AnnotationValidator()
    arguments = sys.argv
    if len(arguments) < 2:
        print("Please specify a video file to annotate!")
        sys.exit(1)
    video_path = arguments[1]
    annotation_output_path = ""
    if len(arguments) == 3:
        annotation_output_path = arguments[2]
    annotation_validator.ReadAnnotations(video_path,annotation_output_path)