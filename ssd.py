#coding=utf-8
# ---------------------------
# @Author	:  Chao Wang
# @Description	:  SSD 的网络结构
# ---------------------------

from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

SSDparams = namedtuple('SSDparams', ['image_shape', #输入图片的大小
	'num_classes',	# 类别： 20+1（1是背景）
	'no_annotation_label',	# 
	'feature_layers',	# 
	'feature_shapes',	# 
	'anchor_size_bounds',
	'anchor_sizes',
	'anchor_ratios',
	'anchor_steps',
	'anchor_offset',
	'normalization',
	'prior_scaling'])

class SSD(object):
	"""docstring for SSD"""
	def __init__(self, is_training=True):
		#super(SSD, self).__init__()
		self.is_training = is_training
		self.threshold = 0.5
		self.ssd_params = SSDparams(image_shape=(300, 300),
									num_classes=2,
									no_annotation_label=2,
									feature_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
									feature_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
									anchor_size_bounds=[0.15, 0.9],
									anchor_sizes=self.get_anchor_sizes(),
									anchor_ratios=[[2, .5], [2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3], [2, .5], [2, .5]],
                                    anchor_steps=[8, 16, 32, 64, 100, 300],
                                    anchor_offset=0.5,
                                    normalizations=[20, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])

	def _build_net(self):
		# 记录detection layers输出
		self.end_points = {}
		# 输入图片的占位节点（固定大小的占位）
		self._images = tf.placeholder(tf.float32,
									 shape=[None, self.ssd_params.image_shape[0], self.ssd_params.image_shape[1], 3])
		# block1
		net = slim.repeat(self._images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		self.end_points['block1'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool1')

		# block2
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		self.end_points['block2'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool2')

		# block3
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		self.end_points['block3'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool3')

		# block4
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		self.end_points['block4'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool4')

		# block5
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')#max pool

        # =============================================================
        # (2) 外接的SSD层
        # =============================================================

        # Block 6: let's dilate the hell out of it!
        # 输出shape为19×19×1024
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net

        # Block 7: 1x1 conv.
        # 卷积核为1×1
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3')
        end_points[end_point] = net

        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3')
        end_points[end_point] = net

        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net




	
	def get_anchor_sizes(self):
		anchor_size_bounds_min, anchor_size_bounds_max = self.ssd_params.anchor_size_bounds[0], self.ssd_params.anchor_size_bounds[1]
		for i in range(1, len(self.ssd_params.feature_layers)+1):
			s[i] = (anchor_size_bounds_min*100 + np.floor(anchor_size_bounds_max+anchor_size_bounds_min/4)*(i-1))/100
		s[0] = 0.5*s[1]
		anchors = s*self.ssd_params.image_shape[0]
		for i in range(len(anchors)-1):
			anchor_sizes[i] = (anchors[i], anchors[i+1])
		return anchor_sizes





