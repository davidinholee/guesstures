import torch
import os
from resnext import resnet101 
from model import generate_model, _modify_first_conv_layer, _construct_depth_model
from PIL import Image
import numpy as np
from torch.autograd import Variable
from preprocessing import parse_annotations, LABELS, get_label_from_frame

N_EX = 8
N_VID = 20

def read_videos(d, label_dict, start):
    # Read in each frame one by one
    data = np.zeros((N_EX, 3, 1, 112, 112))
    label = 0
    dirs = os.listdir(d)
    i = 0
    for n in dirs[start:]:
        if "jpg" in n:
            img = Image.open(d+"/"+n)
            img = img.resize((112, 112))
            # Image data
            img = np.asarray(img)
            img = np.reshape(img, (1, 3, 1, 112, 112))
            data[i] = img
            # Label
            vid_name = d.split("/")[-1]
            frame_n = int(n.split("_")[-1].split(".")[0])
            fs, ls = label_dict[vid_name]
            label = LABELS.index(get_label_from_frame(fs, ls, frame_n))
            i += 1
        if i == N_EX:
            break
    data = np.swapaxes(data, 0, 2)

    return data, label


m = resnet101(sample_size=112, sample_duration=8, num_classes=13)
m = _modify_first_conv_layer(m,3,3)
weights = torch.load("code/best.pth", map_location=torch.device('cpu'))["state_dict"]
weights_new = {}
for key in weights.keys():
    weights_new[key.replace("module.", "")] = weights[key]
m.load_state_dict(weights_new)
print("Model loaded!")

annos = parse_annotations("data/annotations.txt")
data = np.zeros((N_VID, 3, 8, 112, 112))
for j in range(N_VID):
    datas, label = read_videos("data/1CM1_1_R_#217", annos, j+40)
    data[j] = datas
    
img = Variable(torch.from_numpy(data))
output = m(img.float())

prediction = torch.max(output.data, 1)[1].numpy()
print(prediction)
