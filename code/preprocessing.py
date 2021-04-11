import os
import numpy as np
from keras.preprocessing import image

LABELS = ["D0X", "B0A", "B0B", "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11"]
IMAGE_SIZE = (640, 480)

def parse_annotations(filepath):
    '''
    Takes in the annotations file and produces a dictionary that can be queried 
    for the geusture label corresponding to the video and frame in question.
    '''

    label_dict = {}
    f = open(filepath, "r")
    f.readline()
    lines = f.readlines()
    for line in lines:
        tokens = line.split(",")
        if tokens[0] in label_dict:
            # Append new set of frames and label to end
            label_dict[tokens[0]][0].append(int(tokens[3]))
            label_dict[tokens[0]][1].append(tokens[1])
        else:
            # Video hasn't been added to dict yet
            label_dict[tokens[0]] = ([int(tokens[3])], [tokens[1]])
    
    return label_dict

def get_label_from_frame(frames, labels, frame):
    '''
    Given a list of frame numbers and corresponding labels, finds the label
    of a specific frame.
    '''

    for i in range(len(frames)-1):
        if (frame < frames[i+1]):
            return labels[i]
    return labels[-1]


def read_videos(directories, label_dict):
    '''
    Takes in a list of directories and generates two shuffled arrays,
    the input data of size (n, 640, 480, 3), and the labels of size (n)
    '''

    # Get sizes of directories
    tot_size = 0
    for d in directories:
        tot_size += len(os.listdir(d))

    # Read in each frame one by one
    data = np.zeros((tot_size, 640, 480, 3))
    labels = np.zeros((tot_size))
    i = 0
    for d in directories:
        for n in os.listdir(d):
            img = image.load_img(d+n, target_size=IMAGE_SIZE)
            # Image data
            data[i] = np.asarray(img)
            # Label
            vid_name = d.split("/")[-1]
            frame_n = int(n.split("_")[-1].split(".")[0])
            frames, labels = label_dict[vid_name]
            labels[i] = get_label_from_frame(frames, labels, frame_n)
            i += 1
    
    return data, labels
