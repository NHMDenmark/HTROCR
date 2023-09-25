import os
# Ignore TF info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Predictor(object):
    def __init__(self, params):
        with tf.io.gfile.GFile(params['baseline_model_weight_path'], "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                # op_dict=None,
                producer_op_list=None
            )
        self.graph = graph

    def run(self, img, gpu_device="0"):
        session_conf = tf.compat.v1.ConfigProto()
        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.65, visible_device_list=gpu_device)
        session_conf.gpu_options.visible_device_list = gpu_device
        session_conf.gpu_options.allow_growth=True
        # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.75
        with tf.compat.v1.Session(graph=self.graph, config=session_conf) as sess:
            
            if len(img.shape) == 2:
                img = np.expand_dims(img,2)
            img = np.expand_dims(img,0)
            
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            aPred = sess.run(predictor, feed_dict={x: img})
            return aPred[0,:, :,0]
    def pad(image, h=2):
        w = (image.shape[0]/image.shape[1]) * h
        plt.figure(figsize=(w, h))
        plt.imshow(im)
        plt.axis('off')