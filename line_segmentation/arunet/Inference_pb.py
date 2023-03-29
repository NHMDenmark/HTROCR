from __future__ import print_function, division

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import misc

# Code based on ARU-Net repository by Tobias Gruening
# https://github.com/TobiasGruening/ARU-Net

class Inference_pb(object):
    def __init__(self, params):
        with tf.gfile.GFile(params['arunet_weight_path'], "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
        self.graph = graph

    def run(self, img, gpu_device="0"):
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            # Run validation
            aPred = sess.run(predictor, feed_dict={x: img})
            return aPred[0,:, :,0]