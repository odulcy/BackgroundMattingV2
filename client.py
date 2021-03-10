import cv2
import argparse

from threading import Thread, Lock
from vidgear.gears import NetGear

parser = argparse.ArgumentParser(description='Inference from web-cam')

parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1280, 720))
parser.add_argument('--camera-device', type=int, default=0)
parser.add_argument('--crop-width', type=float, default=1.0)

args = parser.parse_args()
device = args.camera_device

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

width, height = args.resolution
cam = Camera(device_id=args.camera_device, width=width, height=height)

crop_width = args.crop_width
middle = int(cam.width // 2)
crop_start = middle - int((cam.width * crop_width)//2)
crop_end = middle + int((cam.width * crop_width)//2)

# define tweak flags
#options = {"flag": 0, "copy": False, "track": False, "bidirectional_mode"=True}
options = {"bidirectional_mode" : True}

client = NetGear(
    address="192.168.0.28",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# loop over until KeyBoard Interrupted
key = "a"
while True:

    try:
        # read frames from stream
        frame = cam.read()

        # send frame to compute_node
        processed_frame = client.send(frame[:,crop_start:crop_end,:], message=key)

        if not (processed_frame is None):

            cv2.imshow("Output Frame", processed_frame)

            # check for 'q' key if pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        break

# close output window
cv2.destroyAllWindows()

exit()
