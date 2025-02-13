import os

os.environ["ROCBLAS_LAYER"] = "2"
os.environ["HIPBLASLT_LOG_MASK"] = "32"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_cublaslt=true"

import numpy as np
import tensorflow.compat.v1 as tf
import random
import copy

if int(tf.__version__[0]) >= 2:
    tf.disable_v2_behavior()


def CompareCpuAndGpu():
    [batch, seqlen, nh, dim] = [80, 128, 8, 64]
    shape_input = [batch, seqlen, nh, dim]

    input_xla = {}
    input_non_xla = {}

    q_placeholder = tf.placeholder(tf.float32, shape_input)
    input_xla[q_placeholder] = [
        [[[random.uniform(-2, 2) for i in range(dim)] for j in range(nh)] for k in range(seqlen)]
        for t in range(batch)
    ]
    input_non_xla[q_placeholder] = copy.deepcopy(input_xla[q_placeholder])

    k_placeholder = tf.placeholder(tf.float32, shape_input)
    input_xla[k_placeholder] = [
        [[[random.uniform(-2, 2) for i in range(dim)] for j in range(nh)] for k in range(seqlen)]
        for t in range(batch)
    ]
    input_non_xla[k_placeholder] = copy.deepcopy(input_xla[k_placeholder])

    def T(t):
        return tf.transpose(t, [0, 2, 1, 3])

    def calc_grad_xla(q, k):
        with tf.xla.experimental.jit_scope(separate_compiled_gradients=True):
            with tf.device("/GPU:0"):
                # q - (batch, nh, seqlen, dim)
                # k - (batch, nh, seqlen, dim)
                qk = tf.matmul(T(q), T(k), transpose_b=True)
        return qk

    def calc_grad_non_xla(q, k):
        with tf.xla.experimental.jit_scope(
            separate_compiled_gradients=True, compile_ops=False
        ):
            with tf.device("/GPU:0"):
                # q - (batch, nh, seqlen, dim)
                # k - (batch, nh, seqlen, dim)
                # qk - (batch, nh, seqlen, seqlen)
                qk = tf.matmul(T(q), T(k), transpose_b=True)
            return qk

    sess_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        res1 = sess.run(
            calc_grad_xla(q_placeholder, k_placeholder),
            feed_dict=input_xla,
        )
        res2 = sess.run(
            calc_grad_non_xla(q_placeholder, k_placeholder),
           feed_dict=input_non_xla,
        )

    return res1, res2


if __name__ == "__main__":
    res1, res2 = CompareCpuAndGpu()
    print("qk: " + str(np.allclose(res1[0], res2[0], rtol=4e-2, atol=4e-2)))
    # print("xla result: ")
    # print(res1)
    # print("non-xla result: ")
    # print(res2)
    # print("diff: ")
    # print(np.abs(res1 - res2))
