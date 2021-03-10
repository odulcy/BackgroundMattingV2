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

import argparse, shutil, time
import cv2
import torch

from PIL import Image
from vidgear.gears import NetGear

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
    def __init__(self, **kwargs):
        self.compute = NetGear(**kwargs)

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

    def close(self):
        """Safely terminates the threads, and NetGear resources."""
        self.compute.close()


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

#torch.cuda.set_device(2)

model = model.cuda().eval()
model.load_state_dict(torch.load(args.model_checkpoint), strict=False)

r,g,b = args.background_color

compute_node = Compute(
    address="127.0.0.1",
    port="5454",
    protocol="tcp",
    pattern=0,
    receive_mode=True,
    logging=True,
    secure_mode=0,
    bidirectional_mode=True,
    THREADED_QUEUE_MODE=True
)

def cv2_frame_to_cuda(frame):
    """Frame as float64"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(frame)
    return img.cuda().div(255).permute((2, 0, 1)).unsqueeze_(0)

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
            if key == ord('q'):
                compute_node.close()
                exit()
        # Matting
        while True:
            # Return processed frame and get a new frame
            data = compute_node.read(processed_frame=frame)
            key, frame = data
            src = cv2_frame_to_cuda(frame)
            pha, fgr = model(src, bgr)[:2]
            res = pha * fgr + (1 - pha) * colored_bgr
            res = res.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            frame = res
            if key == ord('b'):
                break
            if key == ord('q'):
                compute_node.close()
                exit()
