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

from dataset import VideoDataset
from model import MattingBase, MattingRefine
from utils import Displayer

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference from web-cam')

parser.add_argument('--model-type', type=str, required=True, choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)

parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1280, 720))
parser.add_argument('--camera-device', type=int, default=0)
parser.add_argument('--background-color', type=float, nargs=3, metavar=('red', 'green', 'blue'), default=(0,1,0))
parser.add_argument('--crop-width', type=float, default=1.0)

args = parser.parse_args()


# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame
    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


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
width, height = args.resolution
cam = Camera(device_id=args.camera_device, width=width, height=height)

crop_width = args.crop_width
middle = int(cam.width // 2)
crop_start = middle - int((cam.width * crop_width)//2)
crop_end = middle + int((cam.width * crop_width)//2)

dsp = Displayer('MattingV2', int(cam.width * crop_width), cam.height, show_info=(not args.hide_fps))



def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame[:,crop_start:crop_end,:])).unsqueeze_(0).cuda()

with torch.no_grad():
    while True:
        bgr = None
        colored_bgr = None
        while True: # grab bgr
            frame = cam.read()
            key = dsp.step(frame[:,crop_start:crop_end,:])
            if key == ord('b'):
                bgr = cv2_frame_to_cuda(cam.read())
                colored_bgr = torch.zeros_like(bgr).cuda()
                colored_bgr[:,0,:,:] = r
                colored_bgr[:,1,:,:] = g
                colored_bgr[:,2,:,:] = b
                break
            if key == ord('f'):
                dsp.show_info = not dsp.show_info
            elif key == ord('q'):
                exit()
        while True: # matting
            frame = cam.read()
            src = cv2_frame_to_cuda(frame)
            pha, fgr = model(src, bgr)[:2]
            res = pha * fgr + (1 - pha) * colored_bgr
            res = res.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            key = dsp.step(res)
            if key == ord('b'):
                break
            if key == ord('f'):
                dsp.show_info = not dsp.show_info
            elif key == ord('q'):
                exit()
