import os
import numpy as np
import tensorflow as tf
import source.utils as utils
import whiteboxlayer.layers as lay
import whiteboxlayer.extention as ext

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")

        self.dim_node_feat = kwargs['dim_node_feat']
        self.dim_edge_near = kwargs['dim_edge_near']
        self.dim_edge_feat = kwargs['dim_edge_feat']
        self.num_class = kwargs['num_class']
        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.variables = {}

        node_dummy = tf.zeros((1, self.dim_node_feat), dtype=tf.float32)
        edge_dummy = tf.zeros((1, self.dim_edge_near, self.dim_edge_feat), dtype=tf.float32)
        hood_dummy = tf.zeros((1, self.dim_edge_near, 1), dtype=tf.int32)
        pair_dummy = tf.zeros((1, 3), dtype=tf.int32)
        self.__model = Neuralnet(**kwargs)
        self.__model.forward(\
            node_r=node_dummy, edge_r=edge_dummy, hood_r=hood_dummy, \
            node_l=node_dummy, edge_l=edge_dummy, hood_l=hood_dummy, \
            pair=pair_dummy, verbose=True)
        print("\nNum Parameter: %d" %(self.__model.layer.num_params))

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        ftxt = open("list_parameters.txt", "w")
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            text = "T: " + str(key) + str(self.__model.layer.parameters[key].shape)
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

        conc_func = self.__model.__call__.get_concrete_function(\
            tf.TensorSpec(shape=(1, self.dim_node_feat), dtype=tf.float32), \
            tf.TensorSpec(shape=(1, self.dim_edge_near, self.dim_edge_feat), dtype=tf.float32), \
            tf.TensorSpec(shape=(1, self.dim_edge_near, 1), dtype=tf.int32), \
            tf.TensorSpec(shape=(1, self.dim_node_feat), dtype=tf.float32), \
            tf.TensorSpec(shape=(1, self.dim_edge_near, self.dim_edge_feat), dtype=tf.float32), \
            tf.TensorSpec(shape=(1, self.dim_edge_near, 1), dtype=tf.int32), \
            tf.TensorSpec(shape=(1, 3), dtype=tf.int32))
        self.__get_flops(conc_func)

    def __loss(self, y, y_hat):

        entropy_b = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        entropy = tf.math.reduce_mean(entropy_b)

        return {'entropy_b': entropy_b, 'entropy': entropy}

    @tf.autograph.experimental.do_not_convert
    def step(self, minibatch, iteration=0, training=False):

        node_r, edge_r, hood_r = minibatch['r_vertex'], minibatch['r_edge'], minibatch['r_hood_indices']
        node_l, edge_l, hood_l = minibatch['l_vertex'], minibatch['l_edge'], minibatch['l_hood_indices']
        pair = minibatch['label']
        y = minibatch['label_1hot']

        with tf.GradientTape() as tape:
            logit, y_hat = self.__model.forward(\
                node_r=node_r, edge_r=edge_r, hood_r=hood_r, \
                node_l=node_l, edge_l=edge_l, hood_l=hood_l, \
                pair=pair, verbose=False)
            losses = self.__loss(y=y, y_hat=logit)

        if(training):
            gradients = tape.gradient(losses['entropy'], self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/entropy' %(self.__model.who_am_i), losses['entropy'], step=iteration)

        return {'y_hat':y_hat, 'losses':losses}

    def __get_flops(self, conc_func):

        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(conc_func)

        with tf.Graph().as_default() as graph:
            tf.compat.v1.graph_util.import_graph_def(graph_def, name='')

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

            flop_tot = flops.total_float_ops
            ftxt = open("flops.txt", "w")
            for idx, name in enumerate(['', 'K', 'M', 'G', 'T']):
                text = '%.3f [%sFLOPS]' %(flop_tot/10**(3*idx), name)
                print(text)
                ftxt.write("%s\n" %(text))
            ftxt.close()

    def save_params(self, model='base'):

        vars_to_save = self.__model.layer.parameters.copy()
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
        ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def loss_l1(self, x, reduce=None):

        distance = tf.math.reduce_mean(\
            tf.math.abs(x), axis=reduce)

        return distance

    def loss_l2(self, x, reduce=None):

        distance = tf.math.reduce_mean(\
            tf.math.sqrt(\
            tf.math.square(x) + 1e-9), axis=reduce)

        return distance

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = "GCN"
        self.dim_node_feat = kwargs['dim_node_feat']
        self.dim_edge_near = kwargs['dim_edge_near']
        self.dim_edge_feat = kwargs['dim_edge_feat']
        self.num_class = kwargs['num_class']
        self.filters = [256, 256, 512, 512]

        self.layer = lay.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, \
        node_r, edge_r, hood_r, \
        node_l, edge_l, hood_l, \
        pair, verbose=False):

        # origin deco: @tf.function
        # @tf.autograph.experimental.do_not_convert
        agg_r = self.__gcn(node=node_r, edge=edge_r, hood=hood_r, name='gcn', verbose=verbose)
        agg_l = self.__gcn(node=node_l, edge=edge_l, hood=hood_l, name='gcn', verbose=verbose)
        logit = self.__clf(receptor=agg_r, ligand=agg_l, pair=pair, name='clf', verbose=verbose)
        y_hat = tf.nn.softmax(logit, name="y_hat") # speeds up training trick

        return logit, y_hat

    def __gcn(self, node, edge, hood, name='gcn', verbose=True):

        if(verbose): print("\n* GCN")

        for idx in range(len(self.filters)):
            node = ext.pipgcn_node_average(layer=self.layer, node=node, c_out=self.filters[idx], \
                batch_norm=False, activation='relu', name='%s-%d' %(name, idx), verbose=verbose)

        return node

    def __clf(self, receptor, ligand, pair, name='clf', verbose=True):

        if(verbose): print("\n* CLF")

        inter = ext.merge_ligand_receptor(layer=self.layer, ligand=ligand, receptor=receptor, pair=pair, verbose=verbose)

        x = self.layer.fully_connected(x=inter, c_out=512, \
                batch_norm=False, activation='relu', name="%s-clf0" %(name), verbose=verbose)
        x = self.layer.fully_connected(x=x, c_out=self.num_class, \
                batch_norm=False, activation=None, name="%s-clf1" %(name), verbose=verbose)

        return x
