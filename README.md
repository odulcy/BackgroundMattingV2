# Real-Time High-Resolution Background Matting

Fork from the repository [Real-Time High-Resolution Background Matting](https://github.com/PeterL1n/BackgroundMattingV2). [You can check their paper here](https://arxiv.org/abs/2012.07810).

## Download

### Model / Weights

* [Download model / weights](https://drive.google.com/drive/folders/1cbetlrKREitIgjnIikG1HdM4x72FtgBh?usp=sharing)

## Demo

#### Scripts

I mainly modified the `inference_webcam.py` script to use along with Open Broadcaster Software (OBS).
An example of how to use this script is provided in `run.sh`.

&nbsp;

#### Compute remotely matting

I added a proof of concept to offload matting on another computer using the Bidirectional Mode from [vidgear](https://github.com/abhiTronix/vidgear). You can see [more here](https://abhitronix.github.io/vidgear/latest/gears/netgear/advanced/bidirectional_mode/#using-bidirectional-mode-for-video-frames-transfer).

A *Compute Node* is a computer with a GPU and a Client is a computer with a webcam.
The computer node process the image coming from the webcam and send it back to this computer.

You need to adjust IP address in the ``compute_node.py`` and ``client.py``.

**Server part :**
```bash
python compute_node.py --model-backbone-scale 0.25 --model-type mattingrefine --model-backbone resnet50 --model-checkpoint pytorch_resnet50.pth
```

**Client part :**
```bash
python client.py
```

&nbsp;

## Acknowledgements
* [Shanchuan Lin](https://www.linkedin.com/in/shanchuanlin/)*, University of Washington
* [Andrey Ryabtsev](http://andreyryabtsev.com/)*, University of Washington
* [Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/), University of Washington
* [Brian Curless](https://homes.cs.washington.edu/~curless/), University of Washington
* [Steve Seitz](https://homes.cs.washington.edu/~seitz/), University of Washington
* [Ira Kemelmacher-Shlizerman](https://sites.google.com/view/irakemelmacher/), University of Washington

<sup>* Equal contribution.</sup>

&nbsp;

## License ##
This work is licensed under the [MIT License](LICENSE). 
&nbsp;
