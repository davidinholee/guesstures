import os
import numpy as np
from PIL import Image

LABELS = ["D0X", "B0A", "B0B", "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11"]
IMAGE_SIZE = (240, 320)
IMAGE_CHANNELS = 3
CLASS_NUM = 14
VIDEO_BATCH = 5

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


def read_videos(directories, label_dict, count):
    '''
    Takes in a list of directories and generates two shuffled arrays,
    the input data of size (n, 640, 480, 3), and the labels of size (n)
    '''

    # Get sizes of directories
    tot_size = 0
    for d in directories:
        tot_size += (len(os.listdir(d)) - 1)

    # Read in each frame one by one
    data = np.zeros((tot_size, 240, 320, 3))
    labels = np.zeros((tot_size))
    i = 0
    for d in directories:
        for n in np.random.shuffle(os.listdir(d)):
            if "jpg" in n:
                img = Image.open(d+"/"+n)
                # Image data
                data[i] = np.asarray(img)
                # Label
                vid_name = d.split("/")[-1]
                frame_n = int(n.split("_")[-1].split(".")[0])
                fs, ls = label_dict[vid_name]
                labels[i] = LABELS.index(get_label_from_frame(fs, ls, frame_n))
                i += 1

    return data, labels, count

def prepare_data(batch_n):
    '''
    Preprocesses the data we want for training.
    '''

    annos = parse_annotations("data/annotations.txt")
    dirs = [x[0].replace("\\", "/") for x in os.walk("data")]
    if len(dirs) - 1 > batch_n * VIDEO_BATCH:
        return read_videos(dirs[batch_n*VIDEO_BATCH+1:(batch_n+1)*VIDEO_BATCH+1], annos, batch_n+1)
    else:
        return read_videos(dirs[batch_n*VIDEO_BATCH+1:], annos, batch_n+1)

if __name__ == "__main__":
    annos = parse_annotations("data/annotations.txt")
    data, labels = read_videos(["data/1CM1_1_R_#217"], annos)
