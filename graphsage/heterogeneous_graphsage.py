import tensorflow as tf

def sample(self, inputs_u, inputs_l, layer_infos, batch_size=None):
    if batch_size is None:
        batch_size = self.batch_size
    samples_u = [inputs_u]
    samples_l = [inputs_l]
    support_size_u = batch_size
    support_sizes_u = [support_size_u]
    support_size_l = batch_size
    support_sizes_l = [support_size_l]
    for k in range(len(layer_infos)):
        # we must have num_samples_uu == num_samples_ul == num_samples_lu == num_samples_ll
        t = len(layer_infos) - k - 1
        # sample users-to-users
        sampler = layer_infos[t].neigh_sampler_uu
        # node, weights = sampler((samples_u[k], layer_infos[t].num_samples_uu))
        node = sampler((samples_u[k], layer_infos[t].num_samples_uu))
        # weights_uu = tf.reshape(node, [support_size_u * (layer_infos[t].num_samples_uu),])
        sampled_uu = tf.reshape(node, [support_size_u * (layer_infos[t].num_samples_uu),])
        # sample locations-to-users
        sampler = layer_infos[t].neigh_sampler_lu
        node = sampler((samples_l[k], layer_infos[t].num_samples_lu))
        sampled_lu = tf.reshape(node, [support_size_l * (layer_infos[t].num_samples_lu),])
        samples_u.append(tf.concat(sampled_uu, sampled_lu))

        # sample locations-to-locations
        sampler = layer_infos[t].neigh_sampler_ll
        node = sampler((samples_l[k], layer_infos[t].num_samples_ll))
        sampled_ll = tf.reshape(node, [support_size_l * (layer_infos[t].num_samples_ll),])
        # sample users-to-locations
        sampler = layer_infos[t].neigh_sampler_ul
        node = sampler((samples_l[k], layer_infos[t].num_samples_ul))
        sampled_ul = tf.reshape(node, [support_size_u * (layer_infos[t].num_samples_ul),])
        samples_l.append(tf.concat(sampled_ll, sampled_ul))

        support_size_u = support_size_u * layer_infos[t].num_samples_uu + \
                         support_size_l * layer_infos[t].num_samples_lu
        support_size_l = support_size_u * layer_infos[t].num_samples_ul + \
                         support_size_l * layer_infos[t].num_samples_ll
        support_sizes_u.append(support_size_u)
        support_sizes_l.append(support_size_l)
    return samples_u, samples_l, support_sizes_u, support_size_l


def aggregate(self, samples_u, samples_v, input_features, dims, num_samples, support_sizes_u, support_size_v,
         batch_size=None, aggregators_u=None, aggregators_l=None, name=None, concat=False, model_size="small"):

    if batch_size is None:
        batch_size = self.batch_size

    hidden_u = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples_u]
    hidden_l = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples_v]
    new_agg = aggregators is None
    # aggregators is not provided, generate new aggregators
    if new_agg:
        aggregators_u = []
        aggregators_l = []
    for layer in range(len(num_samples)):
        if new_agg:
            dim_mult = 2 if concat and (layer != 0) else 1
            if layer == len(num_samples) - 1:
                aggregator_u = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1], act=lambda x : x,
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
                aggregator_l = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1], act=lambda x : x,
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)                
            else:
                aggregator_u = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1],
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
                aggregator_l = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1],
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
            aggregators_u.append(aggregator_u)
            aggregators_l.append(aggregator_l)
        else:
            aggregator_u = aggregators_u[layer]
            aggregator_l = aggregators_v[layer]
        next_hidden_u = []
        next_hidden_l = []
        for hop in range(len(num_samples) - layer):
            dim_mult = 2 if concat and (layer != 0) else 1
            neigh_dims_u = [support_sizes_u[hop], 
                           num_samples[len(num_samples) - hop - 1], 
                           dim_mult * dims[layer]]
            neigh_dims_l = [support_sizes_l[hop], 
                            num_samples[len(num_samples) - hop - 1], 
                           dim_mult * dims[layer]]
            # aggregator_u(hidden_u[hop],
            #       tf.reshape(hidden_u[hop + 1], neigh_dims),
            #       tf.reshape(weights_u[hop + 1], (support_sizes_u[hop], num_samples[len(num_samples) - hop - 1])))
            h_u = aggregator_u((hidden_u[hop],
                    tf.reshape(hidden_u[hop + 1], neigh_dims)),
                    tf.reshape(hidden_l[hop + 1], neigh_dims))
            h_l = aggregator_u((hidden_l[hop],
                    tf.reshape(hidden_u[hop + 1], neigh_dims)),
                    tf.reshape(hidden_l[hop + 1], neigh_dims))
            next_hidden_u.append(h_u)
            next_hidden_l.append(h_l)
        hidden_u = next_hidden_u
        hideen_l = next_hidden_l
    return hidden_u[0], hidden_l[0], aggregators


class MeanAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            # number of features for neighbours is the same as input
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights_u'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights_u')
            self.vars['neigh_weights_l'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights_l')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs_u, neigh_vecs_l, weights_u, weights_v = inputs

        # dropout
        neigh_vecs_u = tf.nn.dropout(neigh_vecs_u, 1 - self.dropout)
        neigh_vecs_l = tf.nn.dropout(neigh_vecs_l, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        weights_u = tf.expand_dims(weights_u, axis=-1)
        weights_u = tf.tile(weights_u, tf.constant([1, 1, self.input_dim], dtype=tf.int32))
        weights_l= tf.expand_dims(weights_l, axis=-1)
        weights_l = tf.tile(weights_l, tf.constant([1, 1, self.input_dim], dtype=tf.int32))
        neigh_vecs_u *= weights_u
        neigh_vecs_l *= weights_l
        neigh_means_u = tf.reduce_mean(neigh_vecs_u, axis=1)
        neigh_means_l = tf.reduce_mean(neigh_vecs_l, axis=1)

        # [nodes] x [out_dim]
        # (dim_mult * dims[layer], output_dim)
        from_neighs_u = tf.matmul(neigh_means_u, self.vars['neigh_weights_u'])
        from_neighs_l = tf.matmul(neigh_means_l, self.vars['neigh_weights_l'])
        from_neighs = tf.add_n([from_neighs_u, from_neighs_l])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


def _skipgram_loss(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, neg_samples_1, neg_samples_2, hard_neg_samples=None):
    aff_1 = self.affinity(inputs1, inputs2)
    aff_2 = self.affinity(inputs1, inputs3)
    neg_aff = self.neg_cost(inputs1, neg_samples_1, hard_neg_samples)
    # aggeragate negative cost of all negative sampels for each node in batch1
    neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
    loss = tf.reduce_sum(aff_1 + aff_2 - neg_cost)

    aff_1 = self.affinity(inputs4, inputs5)
    aff_2 = self.affinity(inputs4, inputs6)
    neg_aff = self.neg_cost(inputs4, neg_samples_2, hard_neg_samples)
    # aggeragate negative cost of all negative sampels for each node in batch1
    neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
    loss += tf.reduce_sum(aff_1 + aff_2 - neg_cost)
    return loss


class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        # ids are node idx, num_samples is neighbours to sample for each node
        ids, num_samples = inputs
        # use embedding_lookup to get adj lists for nodes needed to sample
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        # the tensor is shuffled along dimension 0 for tf.random_shuffle
        # so have to transpose first
        adj_lists_reshaped = tf.expand_dims(adj_lists, -1)
        weights_reshaped = tf.expand_dims(adj_lists, -1)
        # (batch_size, max_degree, 2)
        adj_weights_lists = tf.concat([adj_lists_reshaped, weights_reshaped], axis=2)
        adj_weights_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists, perm=[1, 0, 2])), perm=[1, 0, 2])

        adj_lists = adj_weights_list[:, :, 0]
        weights_lists = adj_weights_list[:, :, 1]
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        weights_lists = tf.slice(weights_lists, [0,0], [-1, num_samples])
        # return a 2-D tensor
        return adj_lists, weights_lists
