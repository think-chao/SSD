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

        # class和location的预测值
        predictions = []
        locations = []
        for i, layer in enumerate(self.ssd_params.feature_layers):
        	loc, cls = ssd_multibox_layer(self.end_points[layer], self.ssd_params.num_classes
        								  self.ssd_params.anchor_sizes[i], self.ssd_params.anchor_ratios[i],
        								  self.ssd_params.normalizations[i], scope=layer+'_box')
        	predictions.append(slim.softmax(cls))
        	locations.append(loc)
        



	
	def get_anchor_sizes(self):
		anchor_size_bounds_min, anchor_size_bounds_max = self.ssd_params.anchor_size_bounds[0], self.ssd_params.anchor_size_bounds[1]
		for i in range(1, len(self.ssd_params.feature_layers)+1):
			s[i] = (anchor_size_bounds_min*100 + np.floor(anchor_size_bounds_max+anchor_size_bounds_min/4)*(i-1))/100
		s[0] = 0.5*s[1]
		anchors = s*self.ssd_params.image_shape[0]
		for i in range(len(anchors)-1):
			anchor_sizes[i] = (anchors[i], anchors[i+1])
		return anchor_sizes


	def ssd_multibox_layer(x, num_classes, sizes, ratios, normalizations=-1, scope='multibox'):
		# @x: 各层feature map的输出 (例如 None*38*38*128)
		# @numclasses: 类别
		# @sizes: reference box的大小
		# @ratios： 变化的比例
		# @normalization: 是否normalization, -1则否

		pre_shape = [-1] + x.get_shape().as_list()[1:-1]
		with tf.variable_scope(scope):
			if normalizations>0:
				x = self.l2_norm(x, normalization)
				print(x)
			n_anchors = len(sizes) + len(ratios)

			loc_pred = slim.conv2d(x, num_outputs=4*n_anchors, kernel_size=[3, 3], activation_fn=None, scope='conv_loc')
			loc_pred = tf.reshape(loc_pred, pre_shape+[n_anchors, 4])

			cls_pred = slim.conv2d(x, num_outputs=n_anchors*num_classes, kernel_size=[3, 3], activation_fn=None, scope='con_cls')
			cls_pred = tf.reshape(cls_pred, pre_shape+[n_anchors, num_classes])

			return loc_pred, cls_pred



	def l2norm(x, scale, trainable=True, scope='L2Normalization'):
		n_channels = x.get_shape().as_list()[-1]
		l2_norm = tf.nn.l2_normalize(x, axis=3, epsilon=1e-12)
		with tf.variable_scope(scope):
			gamma = tf.get_variable('gamma', shape=[n_channels,], dtype=tf.float32,
									initializer=tf.constant_initializer(scale),
									trainable=trainable)
		return l2_norm*gamma







