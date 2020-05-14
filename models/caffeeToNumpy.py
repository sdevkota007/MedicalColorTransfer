# loads caffe model and tries to save it to numpy ...

import numpy as np
import caffe


print("[INFO] loading model...")
prototxt = "vgg19_gray/deploy.prototxt"
model = "vgg19_gray/vgg19_bn_gray_ft_iter_150000.caffemodel"


def shai_net_to_py_readable(prototxt_filename, caffemodel_filename):
    net = caffe.Net(prototxt_filename, 1, weights=caffemodel_filename) # read the net + weights
    pynet_ = []
    for li in range(len(net.layers)):  # for each layer in the net
        layer = {}  # store layer's information
        layer['name'] = net._layer_names[li]
        # for each input to the layer (aka "bottom") store its name and shape
        layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                             for bi in list(net._bottom_ids(li))]
        # for each output of the layer (aka "top") store its name and shape
        layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                          for bi in list(net._top_ids(li))]
        layer['type'] = net.layers[li].type  # type of the layer
        # the internal parameters of the layer. not all layers has weights.
        layer['weights'] = [net.layers[li].blobs[bi].data[...]
                            for bi in range(len(net.layers[li].blobs))]
        pynet_.append(layer)

    return pynet_

if __name__ == '__main__':
    shai_net_to_py_readable(prototxt, model)