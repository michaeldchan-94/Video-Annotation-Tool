import os
import sys
from typing import Any, List
import cv2
from numpy import dtype, floating, integer, ndarray
from utils.Annotator import Annotator
from utils.utils import apply_infobar, get_optimal_window_scaling, get_scaled_image

class VideoAnnotator:
    """
    Video annotator main class, on StartAnnotation call, will bring up an opencv window
    displaying current allowed keyboard navigation bindings

    Users will then proceed through annotating the given video, first labeling the object they desire to
    track, and then either accepting or fixing the tracking as the video goes on

    Output is written to the given output directory under <video_name>.annotations
    If no output directory is given, default output directory is set to the directory of the passed in video file
    """
    def __init__(self):
        self.frame_number : int = 0
        self.width : int
        self.height : int
        self.normal_keyboard_options : List[str] = ["L/l : Label","S/s : Skip","I/i : Invisible", "Q/q : Quit"]
        self.label_keyboard_options : List[str] = ["Space/Enter : Accept","C/c : Cancel"]
        self.predicted_keyboard_options : List[str] = ["A/a : Accept","F/f : Fix","S/s : Skip","I/i : Invisible", "Q/q : Quit"]
        self.tracking = False
        self.annotator : Annotator
        self.get_next_frame : bool = True
        self.window_scale = 1

        # I could make a Kalman filter here or use like an optical flow approach for tracking, but 
        # For sake of time, let's just use an inbuilt opencv tracker here
        self.tracker : cv2.TrackerCSRT = cv2.TrackerCSRT.create()
        self.frame_copy : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self.cur_frame : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None

    def StartAnnotations(self, video_path : str, annotation_path : str = "") -> None:
        full_video_name : str = os.path.basename(video_path)
        video_name : str = os.path.splitext(full_video_name)[0]

        if annotation_path != "":
            # if an annotation path is provided, use that path
            self.annotator = Annotator(annotation_path)
        else:
            # otherwise, put it in same location as video
            directory_path = os.path.dirname(video_path)
            # Add the ".annotations" extension
            video_name += ".annotations"
            annotation_path = os.path.join(directory_path,video_name)
            self.annotator = Annotator(annotation_path)

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            # Only get the next frame if appropriate key bindings have been pressed
            if (self.get_next_frame):
                ret, frame = cap.read()   
                if not ret:
                    print("End of video.")
                    break

                if self.frame_number == 0:
                    self.window_scale = get_optimal_window_scaling(frame.shape[1],frame.shape[0])
                    print(self.window_scale)
                    frame = get_scaled_image(frame,self.window_scale)
                    self.width = int(frame.shape[1])
                    self.height = int(frame.shape[0])
                else:
                    frame = get_scaled_image(frame,self.window_scale)
                # Make a clean copy in case we need to go back to clean image
                self.frame_copy = frame.copy()

            predicted_enable : bool = False
            # Copy in clean frame copy
            self.cur_frame = self.frame_copy.copy()
            if (self.tracking):
                ok, bbox = self.tracker.update(self.cur_frame)
                if (ok):
                    bbox_lower = (int(bbox[0]), int(bbox[1]))
                    bbox_upper = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    self.cur_frame = cv2.rectangle(self.cur_frame, bbox_lower, bbox_upper, (0, 255, 0), 2)
                    apply_infobar(self.cur_frame ,self.predicted_keyboard_options,self.frame_number,self.width)
                    predicted_enable = True
                    cur_prediction : tuple[int,int,int,int] = bbox
                else:
                    apply_infobar(self.cur_frame ,self.normal_keyboard_options,self.frame_number,self.width)
            else:
                apply_infobar(self.cur_frame ,self.normal_keyboard_options,self.frame_number,self.width)

            cv2.imshow('Video Stream', self.cur_frame)

            self.get_next_frame = False
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif (key == ord('l') or key == ord('L')) and not predicted_enable:
                self.onLabel()
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('s') or key == ord('S')):
                self.onSkip()
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('i') or key == ord('I')):
                self.onInvisible()
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('f') or key == ord('F')) and predicted_enable:
                self.onLabel()
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('a') or key == ord('A')) and predicted_enable:
                self.onAccept(cur_prediction)
                self.get_next_frame = True
                self.frame_number += 1


    def onLabel(self) -> None:
        """
        Allows the user to label the current frame
        User clicks on objects center, and then drags to define the bounding box
        resulting bounding box is calculated based on center and drag coordinates

        This is also the callback for OnFix, as it's the same functionality
        """
        frame = self.frame_copy.copy()
        apply_infobar(frame, self.label_keyboard_options,self.frame_number,self.width)
        x, y, width, height = cv2.selectROI("Video Stream", frame,fromCenter=True,showCrosshair=False)
        self.tracker.init(frame, (x, y, width, height))
        self.tracking = True
        center_x : int = x+int(width/2)
        center_y : int = y+int(height/2)
        self.annotator.write_bounding_box(x_center=center_x,y_center=center_y,width=width,height=height)

    def onSkip(self) -> None:
        """
        Skips the current frame, moving to the next frame
        """
        self.annotator.write_skipped()
    
    def onInvisible(self) -> None:
        """
        Marks the object as invisible in scene, and moves to next frame
        """
        self.annotator.write_invisible()
    
    def onAccept(self,predicted_bbox : tuple[int,int,int,int]) -> None:
        """
        Takes in the current predicted bounding box and writes it to the annotator
        """
        x,y,width,height = predicted_bbox
        center_x : int = x+int(width/2)
        center_y : int = y+int(height/2)
        self.annotator.write_bounding_box(x_center=center_x,y_center=center_y,width=width,height=height)


if __name__ == "__main__":
    video_annotator = VideoAnnotator()
    arguments = sys.argv
    if len(arguments) < 2:
        print("Please specify a video file to annotate!")
        sys.exit(1)
    video_path = arguments[1]
    annotation_output_path = ""
    if len(arguments) == 3:
        annotation_output_path = arguments[2]
    video_annotator.StartAnnotations(video_path,annotation_output_path)