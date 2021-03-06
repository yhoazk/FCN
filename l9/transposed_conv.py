"""
Transposed Convolution Quiz

Transposed convolutions are used to upsample the input and are a core part of the FCN architecture.

In TensorFlow, the API tf.layers.conv2d_transpose is used to create a transposed convolutional layer. Using this documentation, use tf.layers.conv2d_transpose to apply 2x upsampling in the following quiz.

REF: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d_transpose
"""

import oldtensorflow as tf
import numpy as np


def upsample(x):
    """
    Apply a two times upsample on x and return the result.
    :x: 4-Rank Tensor
    :return: TF Operation
    """
    # TODO: Use `tf.layers.conv2d_transpose`
    return tf.layers.conv2d_transpose(inputs=x,
    num_outputs=3,
    kernel_size=2, #what is the difference between 2 and (2,2) for TF?
    stride=2)
    """"
    kernel_size: A list of length 2 holding the [kernel_height, kernel_width] of of the filters. Can be an int if both values are the same.
    stride: A list of length 2: [stride_height, stride_width]. Can be an int if both strides are the same. Note that presently both strides must have the same value.
    """"



x = tf.constant(np.random.randn(1, 4, 4, 3), dtype=tf.float32)
conv = upsample(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(conv)

    print('Input Shape: {}'.format(x.get_shape()))
    print('Output Shape: {}'.format(result.shape))
