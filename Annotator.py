import os
from typing import List

"""
Annotator class that is responsible for the I/O with the annotation file
"""
class Annotator():
    def __init__(self,output_dir,video_name):
        # Add the ".annotations" extension
        video_name += ".annotations"
        self.output_file : str = os.path.join(output_dir,video_name)
        self.create_annotation_file(self.output_file)
        
    def create_annotation_file(self,output_file) -> None:
        open(output_file, "w")

    def write_bounding_box(self,x_center : int, y_center : int, width : int, height : int) -> None:
        txt_str : str = f'V {x_center} {y_center} {width} {height}\n'
        with open(self.output_file, "a") as file:
            file.write(txt_str)

    def write_skipped(self) -> None:
        txt_str : str = f'S -1 -1 -1 -1\n'
        with open(self.output_file, "a") as file:
            file.write(txt_str)

    def write_invisible(self) -> None:
        txt_str : str = f'I -1 -1 -1 -1\n'
        with open(self.output_file, "a") as file:
            file.write(txt_str)