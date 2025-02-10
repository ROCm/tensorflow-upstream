import os

os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./generated "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_html "
)

import numpy as np
import tensorflow.compat.v1 as tf
import random
import copy

if int(tf.__version__[0]) >= 2:
    tf.disable_v2_behavior()


def CompareCpuAndGpu():
    [batch, seqlen, nh, dim] = [80, 128, 8, 64]
    shape_input = [batch, seqlen, nh, dim]
    shape_in_grad = [batch, nh, seqlen, seqlen]
    
    inputs_cpu = {}
    inputs_gpu = {}
    
    q_placeholder = tf.placeholder(tf.float32, shape_input)
    inputs_cpu[q_placeholder] = [
        [
            [[random.uniform(-2, 2) for i in range(dim)] for j in range(nh)]
            for k in range(seqlen)
        ]
        for t in range(batch)
    ]
    inputs_gpu[q_placeholder] = copy.deepcopy(inputs_cpu[q_placeholder])
    
    k_placeholder = tf.placeholder(tf.float32, shape_input)
    inputs_cpu[k_placeholder] = [
        [
            [[random.uniform(-2, 2) for i in range(dim)] for j in range(nh)]
            for k in range(seqlen)
        ]
        for t in range(batch)
    ]
    inputs_gpu[k_placeholder] = copy.deepcopy(inputs_cpu[k_placeholder])
    
    in_grad_placeholder = tf.placeholder(tf.float32, shape_in_grad)
    inputs_cpu[in_grad_placeholder] = [
        [
            [[random.uniform(-2, 2) for i in range(seqlen)] for j in range(seqlen)]
            for k in range(nh)
        ]
        for t in range(batch)
    ]
    inputs_gpu[in_grad_placeholder] = copy.deepcopy(inputs_cpu[in_grad_placeholder])
    
    def T(t):
        return tf.transpose(t, [0, 2, 1, 3])
    
    def calc_grad_cpu(q, k, out_grad):
        with tf.xla.experimental.jit_scope(separate_compiled_gradients=True):
            with tf.device("/CPU:0"):
		        # q - (batch, nh, seqlen, dim)
                # k - (batch, nh, seqlen, dim)
                # qk - (batch, nh, seqlen, seqlen)
                qk = tf.matmul(T(q), T(k), transpose_b=True)
                grad_q, grad_k = tf.gradients(qk, [q, k], out_grad)
            return [qk, grad_q, grad_k]

    def calc_grad_gpu(q, k, out_grad):
        with tf.xla.experimental.jit_scope(separate_compiled_gradients=True):
            with tf.device("/GPU:0"):
		        # q - (batch, nh, seqlen, dim)
                # k - (batch, nh, seqlen, dim)
                # qk - (batch, nh, seqlen, seqlen)
                qk = tf.matmul(T(q), T(k), transpose_b=True)
                # grad_q is correct & grad_k is wrong
                grad_q, grad_k = tf.gradients(qk, [q, k], out_grad) 
                # grad_q is wrong & grad_k is correct
                # grad_q, grad_k = tf.gradients(qk, [q, k], tf.transpose(out_grad, [0, 1, 3, 2]))
            return [qk, grad_q, grad_k]

    sess_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        res1 = sess.run(calc_grad_cpu(q_placeholder, k_placeholder, in_grad_placeholder), feed_dict=inputs_cpu)
        res2 = sess.run(calc_grad_gpu(q_placeholder, k_placeholder, in_grad_placeholder), feed_dict=inputs_gpu)
        
    return res1, res2


if __name__ == "__main__":
    res1, res2 = CompareCpuAndGpu()
    print("qk: " +  str(np.allclose(res1[0], res2[0], rtol=4e-2, atol=4e-2)))
    print("grad_q: " +  str(np.allclose(res1[1], res2[1], rtol=4e-2, atol=4e-2)))
    print("grad_k: " +  str(np.allclose(res1[2], res2[2], rtol=4e-2, atol=4e-2)))

