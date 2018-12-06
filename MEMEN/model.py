import tensorflow as tf
import tensorflow.contrib as tf_contrib

class Memen():
    def __init__(self, l2_lambda, context_len, sentence_len, vocab_size, embedding_size, hidden_size,
                 learning_rate=0.0001, hop_num=3, is_training=True, output_hop_num=3, clip_gradients=5.0):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.l2_lambda = l2_lambda
        self.context_len = context_len
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # system hop number
        self.hop_num = hop_num
        # output hop number
        self.output_hop_num = output_hop_num
        self.is_training = is_training
        print("self.is_training:", self.is_training)

        self.hidden_size = hidden_size

        self.learning_rate = learning_rate
        self.clip_gradients = clip_gradients

        self.query = tf.placeholder(tf.int32, [None, self.sentence_len], name="query")

        self.context = tf.placeholder(tf.int32, [None, self.context_len], name="context")

        # answer start
        self.label_start = tf.placeholder(tf.int32, [None], name="start_point")
        # answer end
        self.label_end = tf.placeholder(tf.int32, [None], name="end_point")

        # initialize the weight

        # word embedding
        self.embedding_word = tf.get_variable("word_embeddings", [self.vocab_size, self.embedding_size])
        self.embedding_query = tf.get_variable("query_embeddings", [self.vocab_size, self.embedding_size])


        self.weight1 = tf.get_variable("weight1", [self.hidden_size * 6, 1])
        self.weight_s = tf.get_variable("weight_s", [self.hidden_size * 2, 1])

        self.start_index, self.end_index = self.inference()
        self.loss_val = self.loss()

        self.predictions_start = tf.argmax(self.start_index, axis=1, name="predictions_start")
        self.predictions_end = tf.argmax(self.end_index, axis=1, name="predictions_end")

        correct_prediction_start = tf.equal(tf.cast(self.predictions_start, tf.int32), self.label_start)
        self.accurate_start = tf.reduce_mean(tf.cast(correct_prediction_start, tf.float32), name="accuracy_start")
        correct_prediction_end = tf.equal(tf.cast(self.predictions_end, tf.int32), self.label_end)
        self.accurate_end = tf.reduce_mean(tf.cast(correct_prediction_end, tf.float32), name="accuracy_end")
        if self.is_training:
            self.train_op = self.train()


    def inference(self):
        # encoding of context and query
        self.encoding_layer()
        # memory network of full-orientation matching
        self.matching_layer()
        # output layer
        start_index, end_index = self.output_layer()
        return start_index, end_index

    def encoding_layer(self):
        # embedding of word
        query_embeddings_word = tf.nn.embedding_lookup(self.embedding_query, self.query)
        context_embeddings_word = tf.nn.embedding_lookup(self.embedding_word, self.context)
        # use BiLSTM to encode both context and query embeddings rnn to encoding inputs(query)
        forward = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        backward = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        query_outputs, query_state = tf.nn.bidirectional_dynamic_rnn(forward, backward, query_embeddings_word,
                                                                     dtype=tf.float32, scope="query")
        # query representation. bi_outputs_query contain two elements.one for forward part, another for backward part
        output_fw = query_outputs[0]
        output_bw = query_outputs[1]
        # [none, sentence_len, hidden_size*2]
        self.rQ = tf.concat([output_fw, output_bw], -1)
        # the concatenation of both directions' last hidden state.
        output_state_fw = query_state[0][1]
        output_state_bw = query_state[1][1]
        # [none, hidden_size*2]
        self.uQ = tf.concat([output_state_fw, output_state_bw], -1)
        # context representation
        context_outputs, context_context = tf.nn.bidirectional_dynamic_rnn(forward, backward, context_embeddings_word,
                                                                           dtype=tf.float32, scope="context")
        output_fw = context_outputs[0]
        output_bw = context_outputs[1]
        # [none, context_len, hidden_size*2]
        self.rP = tf.concat([output_fw, output_bw], -1)

    def matching_layer(self):
        # memory network of full-orientation matching
        with tf.variable_scope("matching_layer"):
            # memory hop operation
            for i in range(self.hop_num):
                if i == 1:
                    tf.get_variable_scope().reuse_variables()
                # integral query matching.
                self.m_1 = self.integral_query_matching()

                # query-based similarity matching.
                self.M_2 = self.query_based_similarity_matching()

                # context-based similarity matching
                self.m_3 = self.context_based_similarity_matching()

                # integrated hierarchical matching
                self.M = self.integrated_hierarchical_matching()

                # add an additional gate
                self.M = self.add_gate()

                # pass through a bi-directional LSTM.
                self.O = self.capture_interaction_context_query()

                # In multiple layers, the integrated hierarchical matching module M can be regarded as the input rP
                # of the next layer after a dimensionality reduction processing
                with tf.variable_scope("dimension_reduction"):
                    self.rP = tf.layers.dense(self.O, self.hidden_size * 2)

    # obtain the importance of each word in passage according to the integral query by means of
    # computing the match between uQ and each representation rP by taking the inner product followed by a softmax
    def integral_query_matching(self):
        # insert a dim [none, 1, hidden_size*2]
        uQ_expand = tf.expand_dims(self.uQ, axis=1)
        # [none, context_len] inner product
        inner_product = tf.reduce_sum(tf.multiply(uQ_expand, self.rP), axis=2)
        # softmax [none, context_len]
        c = tf.nn.softmax(inner_product, dim=1)
        # [none, context_len, 1] sum of context weighted by attention c
        c_expand = tf.expand_dims(c, axis=2)
        # sum of the rP weighted by attention ct
        m_1 = tf.reduce_sum(tf.multiply(c_expand, self.rP), axis=1)
        return m_1

    # obtain an alignment matrix A (n*m) between the query and context by Aij=w1[rP;rQ;rP*rQ]
    # w1 is the weight parameter, * is element-wise multiplication
    def query_based_similarity_matching(self):
        with tf.variable_scope("query_based_similarity_matching"):
            # [none, context_len, 1, hidden_size*2]
            rP_expand = tf.expand_dims(self.rP, axis=2)
            # [none, 1, sentence_len, hidden_size*2]
            rQ_expand = tf.expand_dims(self.rQ, axis=1)
            # [none,article_len,sentence_len]
            r_p_q = tf.multiply(rP_expand, rQ_expand)
            r_p = tf.tile(rP_expand, [1, 1, self.sentence_len, 1])
            r_q = tf.tile(rQ_expand, [1, self.context_len, 1, 1])
            # concat to generate Aij
            A = tf.concat([r_p, r_q, r_p_q], axis=-1)

            self.A = tf.squeeze(tf.layers.dense(A, 1), axis=-1)
            # softmax function is performed across the row vector
            B = tf.nn.softmax(self.A, dim=2)
            # each attention vector [none, context_len, hidden_size*2]
            M_2 = tf.matmul(B, self.rQ)
        return M_2

    # When we consider the relevance between context and query, the most representative word in the query sentence
    # can be chosen by e = max(A), and attention is d = softmax(e)
    def context_based_similarity_matching(self):
        # the most representative word [none,article_len]
        e = tf.reduce_max(self.A, axis=2)
        # attention [none,article_len]
        d = tf.nn.softmax(e, dim=1)
        d_expand = tf.expand_dims(d, axis=-1)
        # last matching module [none, hidden_size*2]
        m_3 = tf.reduce_sum(tf.multiply(d_expand, self.rP), axis=1)
        return m_3

    def integrated_hierarchical_matching(self):
        # linear function to get the integrated hierarchical matching module
        with tf.variable_scope("integrated_hierarchical_matching"):
            # expand dimensions for matrix [none, 1, hidden_size*2]
            m_1 = tf.expand_dims(self.m_1, axis=1)
            m_3 = tf.expand_dims(self.m_3, axis=1)
            # M1 and M3 are matrixes that are tiled n times by m_1 and m_3
            M_1 = tf.tile(m_1, [1, self.context_len, 1])
            M_3 = tf.tile(m_3, [1, self.context_len, 1])

            M_1 = tf.layers.dense(M_1, self.hidden_size * 2)
            M_2 = tf.layers.dense(self.M_2, self.hidden_size * 2)
            M_3 = tf.layers.dense(M_3, self.hidden_size * 2)
            # linear function that could be tuned [none, context_len, hidden_size*2]
            M = 0.5 * M_1 + 0.25 * M_2 + 0.25 * M_3
        return M

    # add an additional gate to the input of RNN
    def add_gate(self):
        # filtrates the part of tokens that are helpful in understanding the relation between passage and query.
        with tf.variable_scope("add_gate"):
            # add bias
            b = 0
            # M = tf.nn.bias_add(M, b)
            gt = tf.sigmoid(tf.layers.dense(self.M, 1))
        return tf.multiply(gt, self.M)

    def capture_interaction_context_query(self):
        with tf.variable_scope("capture_interaction"):
            forward = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            backward = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(forward, backward, self.M, dtype=tf.float32,
                                                             time_major=False, swap_memory=True, scope="capture_interaction")
            # bi_outputs_query contain two elements.one for forward part, another for backward part
            result = tf.concat([outputs[0], outputs[1]], axis=-1)
            # [none, context_len, hidden_size * 2]
        return result

    # we follow Wang and Jiang to use the boundary model of pointer networks to locate the answer in the passage.
    # we follow Wang to initialize the hidden state of the pointer network by a query-aware representation:
    def output_layer(self):
        # initialize the hidden state of the pointer network by a query-aware representation.
        start_prediction = None
        end_prediction = None
        with tf.variable_scope("output_layer"):
            l_0 = self.initial_hidden_state()
            # k = 1,2 represent the start point and end point of the answer
            l_k_1 = l_k_2 = l_0
            for i in range(self.output_hop_num):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                start_prediction, end_prediction, v_k_1, v_k_2 = self.point_network(l_k_1, l_k_2)

                # pass O weighted by current predicted probability aK
                # use GRU to update l_k with v_k as input
                l_k_1 = self.GRU(1, l_k_1, v_k_1)
                l_k_2 = self.GRU(2, l_k_2, v_k_2)
        return start_prediction, end_prediction

    def initial_hidden_state(self):
        # initialize the hidden state of the pointer network by a query-aware representation
        with tf.variable_scope("initial_hidden_state"):
            z_ = tf.layers.dense(self.rQ, self.hidden_size * 2, activation=tf.nn.tanh, use_bias=True)
            z = tf.layers.dense(z_, 1)
            a = tf.nn.softmax(z, dim=1)

            l_0 = tf.reduce_sum(tf.multiply(a, self.rQ), axis=1)
        return l_0

    # it is a output of matching layer.
    def point_network(self, l_k_1, l_k_2):
        with tf.variable_scope("point_network"):
            start, v_k_1 = self.point_network_single(1, l_k_1)
            end, v_k_2 = self.point_network_single(2, l_k_2)
        return start, end, v_k_1, v_k_2

    # point network, single
    def point_network_single(self, k, l_k):
        with tf.variable_scope("point_network_" + str(k)):
            O_j = tf.layers.dense(self.O, self.hidden_size * 2)
            l = tf.expand_dims(tf.layers.dense(l_k, self.hidden_size * 2), axis=1)
            z_j = tf.nn.tanh(O_j + l)

            a = tf.nn.softmax(tf.squeeze(tf.layers.dense(z_j, 1), axis=2), dim=1)

            v_k = tf.reduce_sum(tf.multiply(tf.expand_dims(a, axis=2), self.O), axis=1)
        return a, v_k

    def GRU(self, k, l_k, v_k):
        with tf.variable_scope("GRU_" + str(k)):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size * 2)
            l_k = cell(l_k, v_k)
        return l_k[0]

    def loss(self):
        #label_start:[none]; self.logits_start:[none, article_len]
        loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_start, logits=self.start_index)
        loss_start = tf.reduce_mean(loss_start)
        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_end, logits=self.end_index)
        loss_end = tf.reduce_mean(loss_end)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name)]) * self.l2_lambda
        loss = loss_start+loss_end+l2_losses
        return loss


    def train(self):
        #learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps, self.decay_rate,staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op