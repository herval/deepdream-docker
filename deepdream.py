# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import os
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import time

import caffe

def showarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    millis = int(round(time.time() * 1000))
    filename = "/data/output/tmp/steps-%i.jpg" % millis
    PIL.Image.fromarray(np.uint8(a)).save(filename)

input_file = os.getenv('INPUT', 'input.png')
iterations = os.getenv('ITER', 50)
try:
    iterations = int(iterations)
except ValueError:
    iterations = 50

scale = os.getenv('SCALE', 0.05)
try:
    scale = float(scale)
except ValueError:
    scale = 0.05

model_name = os.getenv('MODEL', 'inception_4c/output')
print "Processing file: " + input_file

img = np.float32(PIL.Image.open('/data/%s' % input_file))

model_path = '/caffe/models/bvlc_googlenet/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    '''Basic gradient ascent step.'''
    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)
        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def verifyModel(net, model):
    print "Verifying model: %s" % model
    keys = net.blobs.keys()
    if model in keys:
        print "Model %s is valid." %model
        return True
    else:
        print "Invalid model: %s.  Valid models are:" % model
        for k in keys:
            print k
        return False

if not verifyModel(net, model_name):
    os._exit(1)

if not os.path.exists("/data/output"):
  os.mkdir("/data/output")

if not os.path.exists("/data/output/tmp"):
  os.mkdir("/data/output/tmp")

print "This might take a little while..."
print "Generating first sample..."
step_one = deepdream(net, img)
PIL.Image.fromarray(np.uint8(step_one)).save("/data/output/step_one.jpg")

print "Generating second sample..."
step_two = deepdream(net, img, end='inception_3b/5x5_reduce')
PIL.Image.fromarray(np.uint8(step_two)).save("/data/output/step_two.jpg")

frame = img
frame_i = 0

h, w = frame.shape[:2]
s = float(scale) # scale coefficient
print "Entering dream mode..."
print "Iterations = %s" % iterations
print "Scale = %s" % scale
print "Model = %s" % model_name
for i in xrange(int(iterations)):
    print "Step %d of %d is starting..." % (i, int(iterations))
    frame = deepdream(net, frame, end=model_name)
    PIL.Image.fromarray(np.uint8(frame)).save("/data/output/%04d.jpg"%frame_i)
    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    frame_i += 1
    print "Step %d of %d is complete." % (i, int(iterations))

print "All done! Check the /output folder for results"