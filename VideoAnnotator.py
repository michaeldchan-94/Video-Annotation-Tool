import os
import sys
from typing import Any, List
import cv2
from numpy import dtype, floating, integer, ndarray
from utils.Annotator import Annotator
from utils.utils import apply_infobar, get_optimal_window_scaling, get_scaled_image
import argparse
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np

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
        self.prompt_keyboard_options : List[str] = ["A/a : Accept","F/f : Fix","L/l : Rerun model","S/s : Skip","I/i : Invisible", "Q/q : Quit"]
        self.tracking = False
        self.annotator : Annotator
        self.get_next_frame : bool = True
        self.window_scale = 1

        # I could make a Kalman filter here or use like an optical flow approach for tracking, but 
        # For sake of time, let's just use an inbuilt opencv tracker here
        self.tracker : cv2.TrackerCSRT = cv2.TrackerCSRT.create()
        self.frame_copy : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self.cur_frame : cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None

        # Prompt based stuff
        self.prompt_enable = False
        self.processor : OwlViTProcessor
        self.model : OwlViTForObjectDetection
        self.prompt_bar = False


    def StartAnnotations(self, video_path : str, annotation_path : str = "", prompt_str : str = "") -> None:
        full_video_name : str = os.path.basename(video_path)
        video_name : str = os.path.splitext(full_video_name)[0]

        if prompt_str != "":
            # Only load in model if prompt is valid
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.prompt_enable = True

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
                    frame = get_scaled_image(frame,self.window_scale)
                    self.width = int(frame.shape[1])
                    self.height = int(frame.shape[0])
                else:
                    frame = get_scaled_image(frame,self.window_scale)
                # Make a clean copy in case we need to go back to clean image
                self.frame_copy = frame.copy()

            predicted_enable : bool = False
            prompt_bar : bool = False
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
            elif self.prompt_enable:
                cur_prediction : tuple[int,int,int,int] = self.run_prompt_model(self.cur_frame,prompt_str)
                prompt_bar = True
            else:
                apply_infobar(self.cur_frame ,self.normal_keyboard_options,self.frame_number,self.width)

            cv2.imshow('Video Stream', self.cur_frame)

            self.get_next_frame = False
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif (key == ord('l') or key == ord('L')) and not predicted_enable and not prompt_bar:
                self.onLabel()
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('l') or key == ord('L')) and not predicted_enable and prompt_bar:
                print("rerunning prompt")
                self.cur_frame = self.frame_copy.copy() #get clean image
                self.run_prompt_model(self.cur_frame,prompt_str)
            elif (key == ord('s') or key == ord('S')):
                self.onSkip()
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('i') or key == ord('I')):
                self.onInvisible()
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('f') or key == ord('F')) and (predicted_enable or prompt_bar):
                self.onLabel()
                if not predicted_enable:
                    self.tracker.init(frame, cur_prediction)
                    self.tracking = True
                self.prompt_enable = False
                self.get_next_frame = True
                self.frame_number += 1
            elif (key == ord('a') or key == ord('A')) and (predicted_enable or prompt_bar):
                self.onAccept(cur_prediction)
                if not predicted_enable:
                    self.tracker.init(frame, cur_prediction)
                    self.tracking = True
                self.prompt_enable = False
                self.get_next_frame = True
                self.frame_number += 1


    def run_prompt_model(self,frame,prompt_str):
        inputs = self.processor(text=[prompt_str], images=self.cur_frame, return_tensors="pt")
        outputs = self.model(**inputs)
        pil_image = Image.fromarray(frame)
        target_sizes = [pil_image.size[::-1]]
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        # print('here')
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        # Get highest scoring box
        if len(scores) == 0:
            return (1,1,1,1)
        score_list = scores.tolist()
        max_prediction_idx = np.argmax(score_list)
        if score_list[max_prediction_idx] > 0.1:
            label = prompt_str
            box_list = boxes.tolist()
            box = [int(i) for i in box_list[max_prediction_idx]]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score_list[max_prediction_idx]:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            apply_infobar(frame ,self.prompt_keyboard_options,self.frame_number,self.width)
            return (box[0], box[1], box[2], box[3])

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
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Path to the input file.')
    parser.add_argument('--output', type=str, default="",
                    help='The path for the output annotation file')
    parser.add_argument('--prompt', type=str, default="",
                    help='The prompt to pass to the model')
    args = parser.parse_args()
    print(args)
    video_annotator.StartAnnotations(args.input_file,args.output,args.prompt)