import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_user_ops


arr0 = np.array([[1,2],[3,4],[5,6]], dtype = np.int32)
np.save("huge0.npy", arr0)
arr1 = np.array([[1,2,3],[4,5,6]], dtype = np.int64)
np.save("huge1.npy", arr1)
arr2 = np.array([[1,2,3,4,5,6]], dtype = np.float32)
np.save("huge2.npy", arr2)
arr3 = np.random.rand(1000000, 64).astype(np.float16)
np.save("huge3.npy", arr3)


g = tf.Graph()


with g.as_default():
#  with tf.device("/CPU:0"):
  
  a0 = gen_user_ops.huge_const(path="huge0.npy", dtype = tf.int32, shape = (3,2))
  a1 = gen_user_ops.huge_const(path="huge1.npy", dtype = tf.int64, shape = (2,3))
  a2 = gen_user_ops.huge_const(path="huge2.npy", dtype = tf.float32, shape = (1,6))
  o0 = tf.add(a0, 0, name = "o0")
  o1 = tf.add(a1, 1, name = "o1")
  o2 = tf.add(a2, 2, name = "o2")
  
  '''
  a3 = gen_user_ops.huge_const(path="huge3.npy", dtype = tf.float16, shape = (1000000,64))
  p3 = tf.placeholder(tf.float16, [None], name="p3")
  o3 = tf.add(a3, p3, name = "o3")
  '''
from google.protobuf import text_format

def save_pbtxt(pb, file_name):
  #with open(file_name, "w") as f:
  with tf.gfile.GFile(file_name, 'w') as f:
    f.write(text_format.MessageToString(pb))

save_pbtxt(g.as_graph_def(), "./huge.pbtxt")

#==================================================================

f = open("./huge.pbtxt", "r")
graph_def = text_format.Parse(f.read(), tf.GraphDef())

# Import the graph protobuf into our new graph.
g2 = tf.Graph()
with g2.as_default():
    tf.import_graph_def(graph_def=graph_def, name="")

outputs = ["o0:0", "o1:0", "o2:0"]
inputs = {}
#outputs = ["o3:0"]
#inputs = {"p3:0":[1]}
with tf.Session(graph = g2) as sess:
  res = sess.run(outputs, inputs)
  #for i in range(5):
  #  res = sess.run(outputs, inputs)
  for item in res:
    print(item)
