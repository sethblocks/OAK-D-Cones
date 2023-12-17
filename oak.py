from depthai_sdk import OakCamera, ArgsParser
import argparse
from depthai import ColorCameraProperties as cprop
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='model/yolo.json', type=str)
args = ArgsParser.parseArgs(parser)

with OakCamera(args=args) as oak:
    color = oak.create_camera('color', fps=15.0, resolution=cprop.SensorResolution.THE_720_P)
    nn = oak.create_nn(args['config'], color, nn_type="yolo", spatial=True)
    #oak.visualize(color, fps=True, scale=2/3)
    oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)
