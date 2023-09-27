# coding=utf-8
import os

from models import GCN, PTDNetGCN
from tf_geometric.layers import SAGPool, GCN
import tf_geometric as tfg
from tf_geometric.utils import tf_utils
from scipy.sparse import coo_matrix
import scipy.sparse as sp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
import logging
from tqdm import tqdm



# 次方案不支持跨进程的模式，可以选择文件名称增加进程ID
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('PTDNet_PROTEINS.log', 'a'))

# 所有的print输出之后全部是info等级的日志，替换函数
print = logger.info

# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
graph_dicts = tfg.datasets.TUDataset("PROTEINS").load_data()

# Since a TU dataset may contain node_labels, node_attributes etc., each of which can be used as node features
# We process each graph as a dict and return a list of dict for graphs
# You can easily construct you Graph object with the data dict

num_node_labels = np.max([np.max(graph_dict["node_labels"]) for graph_dict in graph_dicts]) + 1


def convert_node_labels_to_one_hot(node_labels):
    num_nodes = len(node_labels)
    x = np.zeros([num_nodes, num_node_labels], dtype=np.float32)
    x[list(range(num_nodes)), node_labels] = 1.0
    return x


def construct_graph(graph_dict):
    return tfg.Graph(
        x=convert_node_labels_to_one_hot(graph_dict["node_labels"]),
        edge_index=graph_dict["edge_index"],
        y=graph_dict["graph_label"]  # graph_dict["graph_label"] is a list with one int element
    )


graphs = [construct_graph(graph_dict) for graph_dict in graph_dicts]
num_classes = np.max([graph.y[0] for graph in graphs]) + 1

train_graphs, test_graphs = train_test_split(graphs, test_size=0.1)


def create_graph_generator(graphs, batch_size, infinite=False, shuffle=False):
    while True:
        dataset = tf.data.Dataset.range(len(graphs))
        if shuffle:
            dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size)

        for batch_graph_index in dataset:
            batch_graph_list = [graphs[i] for i in batch_graph_index]

            batch_graph = tfg.BatchGraph.from_graphs(batch_graph_list)
            yield batch_graph

        if not infinite:
            break


batch_size = 512


# SAGPool_h
class SAGPoolHModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcns = []
        self.sag_pools = []

        for _ in range(3):
            self.gcns.append(GCN(128, activation=tf.nn.relu))
            self.sag_pools.append(SAGPool(
                score_gnn=GCN(1),
                ratio=0.5,
                score_activation=tf.nn.tanh
            ))

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None, mask=None):

        x, edge_index, edge_weight, node_graph_index = inputs
        h = x

        outputs = []
        for gcn, sag_pool in zip(self.gcns, self.sag_pools):
            h = gcn([h, edge_index, edge_weight], training=training)
            h, edge_index, edge_weight, node_graph_index = sag_pool([h, edge_index, edge_weight, node_graph_index],
                                                                    training=training)
            output = tf.concat([
                tfg.nn.mean_pool(h, node_graph_index),
                tfg.nn.max_pool(h, node_graph_index)
            ], axis=-1)
            outputs.append(output)

        h = tf.reduce_sum(tf.stack(outputs, axis=1), axis=1)

        # Predict Graph Labels
        h = self.mlp(h, training=training)
        return h


# model = SAGPoolHModel()
model = PTDNetGCN(3,2)

def forward(batch_graph, training=False):
    input_data= coo_matrix(
            (np.ones(batch_graph.num_edges), (batch_graph.edge_index[0], batch_graph.edge_index[1])),
            shape=(batch_graph.num_nodes, batch_graph.num_nodes))
    indices = list(zip(*input_data.nonzero()))
    
    adj=tf.SparseTensor(indices=indices, values=np.float32(input_data.data), dense_shape=input_data.get_shape())
    model.set_fea_adj(np.array(range(adj.shape[1])), batch_graph.x, adj)
    
    temperature =1.0
    # for l in range(3):
    #     output = model.call(temperature,training=True)
    #     preds.append(tf.expand_dims(output,0))
    # all_preds = tf.concat(preds,axis=0)
    # # mean_preds = tf.reduce_mean(preds,axis=0)
    output = model.call(temperature,training=True)
    node_graph_index = batch_graph.node_graph_index 
    # Mean Pooling
    h = tfg.nn.mean_pool(output, node_graph_index)

    return  h


def evaluate():
    accuracy_m = tf.keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2) #default 1e-3

train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)


toal_accuracy=[]
for step in tqdm(range(10000)):
    train_batch_graph = next(train_batch_generator)
    with tf.GradientTape() as tape:
        logits = forward(train_batch_graph, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
        )

    vars = tape.watched_variables()
    grads = tape.gradient(losses, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 1 == 0:
        mean_loss = tf.reduce_mean(losses)
        accuracy = evaluate()
        toal_accuracy.append(accuracy)
        print(("step = {}\tloss = {}\taccuracy = {}".format(step, mean_loss, accuracy)))
mean=np.mean(toal_accuracy[-10:])
std_dev=np.std(toal_accuracy[-10:])
print(("mean = {}\tstd = {}\t".format(mean, std_dev)))