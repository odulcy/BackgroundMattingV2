"""
Inference on webcams: Use a model on webcam input.

Once launched, the script is in background collection mode.
Press B to toggle between background capture mode and matting mode. The frame shown when B is pressed is used as background for matting.
Press Q to exit.

Example:

    python inference_webcam.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --resolution 1280 720

"""

import argparse, os, shutil, time
import cv2
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
from threading import Thread, Lock
from tqdm import tqdm
from PIL import Image
from vidgear.gears import NetGear

from dataset import VideoDataset
from model import MattingBase, MattingRefine

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference from web-cam')

parser.add_argument('--model-type', type=str, required=True, choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)

parser.add_argument('--background-color', type=float, nargs=3, metavar=('red', 'green', 'blue'), default=(0,1,0))

args = parser.parse_args()


# ----------- Utility classes -------------

class Compute:
    def __init__(self):
        # define tweak flags
        #options = {"flag": 0, "copy": False, "track": False, "bidirectional_mode": True}
        options = {"bidirectional_mode": True}

        self.compute = NetGear(
            address="192.168.0.28",
            port="5454",
            protocol="tcp",
            pattern=1,
            receive_mode=True,
            logging=True,
            **options
        )

    def read(self, processed_frame=None):
        """Read data from client and send back a processed frame
        Args:
            processed_frame (NumPy array)
        Returns:
            data : server_data, frame (string, NumPy array).

        Note : server_data can be any type, but here it will be
            only a string
        """
        data = self.compute.recv(return_data=processed_frame)
        return data


# --------------- Main ---------------


# Load model
if args.model_type == 'mattingbase':
    model = MattingBase(args.model_backbone)
if args.model_type == 'mattingrefine':
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold)

model = model.cuda().eval()
model.load_state_dict(torch.load(args.model_checkpoint), strict=False)

r,g,b = args.background_color

compute_node = Compute()

def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()

frame = None
with torch.no_grad():
    while True:
        bgr = None
        colored_bgr = None
        # Grab background
        while True:
            data = compute_node.read(frame)
            if data is None:
                break
            key, frame = data
            if key == ord('b'):
                bgr = cv2_frame_to_cuda(frame)
                colored_bgr = torch.zeros_like(bgr).cuda()
                colored_bgr[:,0,:,:] = r
                colored_bgr[:,1,:,:] = g
                colored_bgr[:,2,:,:] = b
                break
        # Matting
        while True:
            # Return processed frame and get a new frame
            data = compute_node.read(processed_frame=frame)
            if data is None:
                break
            key, frame = data
            src = cv2_frame_to_cuda(frame)
            pha, fgr = model(src, bgr)[:2]
            res = pha * fgr + (1 - pha) * colored_bgr
            res = res.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            frame = res
            if key == ord('b'):
                break
