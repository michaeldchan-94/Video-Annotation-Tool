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


from utils import apply_infobar, read_annotations,Annotation


class AnnotationValidator:
    def __init__(self):
        self.annotations : List[Annotation]
        self.frame_copy : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self.cur_frame : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self.normal_keyboard_options : List[str] = ["n : Next Frame","N : Forward 10 Frames","p : Previous Frame", "P : Back 10 Frames"]
        self.get_next_frame : bool = True
        self.frame_number : int = 0

    def ReadAnnotations(self, video_path : str, annotation_path : str = ""):
        full_video_name : str = os.path.basename(video_path)
        video_name : str = os.path.splitext(full_video_name)[0]

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
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.onNextFrame()
                    self.get_next_frame = True
                    self.frame_number += 1
                elif key == ord('N'):
                    self.onForward10()
                    self.get_next_frame = True
                    self.frame_number += 1
                elif key == ord('p'):
                    self.onPreviousFrame()
                    self.get_next_frame = True
                    self.frame_number += 1
                elif key == ord('P'):
                    self.onBack10()
                    self.get_next_frame = True
                    self.frame_number += 1
    
    def onNextFrame(self):
        pass
    def onForward10(self):
        pass
    def onPreviousFrame(self):
        pass
    def onBack10(self):
        pass

    def __apply_annotation(self, frame : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]],
                           annotation : Annotation):
        bottom_x : int = annotation.center_x-int(annotation.width/2)
        bottom_y : int = annotation.center_y-int(annotation.height/2)
        upper_x : int = annotation.center_x+int(annotation.width/2)
        upper_y : int = annotation.center_y+int(annotation.height/2)
        frame = cv2.rectangle(frame, (bottom_x,bottom_y),(upper_x,upper_y), (0, 255, 0), 2)




if __name__ == "__main__":
    annotation_validator = AnnotationValidator()
    # arguments = sys.argv
    # if len(arguments) < 2:
    #     print("Please specify a video file to annotate!")
    #     sys.exit(1)
    # video_path = arguments[1]
    # annotation_output_path = ""
    # if len(arguments) == 3:
    #     annotation_output_path = arguments[2]

    video_path = "test/853889-hd_1920_1080_25fps.mp4"
    annotation_output_path = "test/853889-hd_1920_1080_25fps.annotations"
    annotation_validator.ReadAnnotations(video_path,annotation_output_path)