from vidgear.gears import NetGear
import cv2

stream = cv2.VideoCapture(0)

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
        (grabbed, frame) = stream.read()

        # check for frame if not grabbed
        if not grabbed:
            break

        # send frame to compute_node
        processed_frame = client.send(frame, message=key)

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

# safely close video stream
stream.release()

# safely close compute_node
client.close()
