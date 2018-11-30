import tensorflow as tf
import tensorflow.contrib as tf_contrib



class Memen():
    def __init__(self, l2_lambda, context_len, sentence_len, vocab_size, embed_size,
                 learning_rate=0.0001, hop_num=3, is_training=True, output_hop_times=3, clip_gradients=5.0):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.l2_lambda = l2_lambda
        self.context_len = context_len
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size


        self.hop_num = hop_num

        self.output_hop_times = output_hop_times
        self.is_training = is_training
        print("self.is_training:", self.is_training)

        self.hidden_size = embed_size

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
        self.embedding_word = tf.get_variable("word_embeddings", [self.vocab_size, self.embed_size])
        self.embedding_query = tf.get_variable("query_embeddings", [self.vocab_size, self.embed_size])


        self.weight1 = tf.get_variable("weight1", [self.hidden_size * 6, 1])
        self.weight_s = tf.get_variable("weight_s", [self.hidden_size * 2, 1])

        self.start_index, self.end_index = self.inference()
        self.loss_val = self.loss()

        self.predictions_start = tf.argmax(self.start_index, axis=1, name="predictions_start")  # shape:(?,)
        self.predictions_end = tf.argmax(self.end_index, axis=1, name="predictions_end")  # shape:(?,)

        correct_prediction_start = tf.equal(tf.cast(self.predictions_start, tf.int32), self.label_start)
        self.accurate_start = tf.reduce_mean(tf.cast(correct_prediction_start, tf.float32), name="accuracy_start")  # shape=()
        correct_prediction_end = tf.equal(tf.cast(self.predictions_end, tf.int32), self.label_end)
        self.accurate_end = tf.reduce_mean(tf.cast(correct_prediction_end, tf.float32), name="accuracy_end")  # shape=()
        if self.is_training:
            self.train_op = self.train()


    def inference(self):
        # 1.encoding of context and query
        self.encode_layer()
        # 2.memory network of full-orientation matching
        self.matching_layer()
        # 3.output layer
        start_index, end_index = self.output_layer()
        return start_index, end_index

    def encode_layer(self):
        # 1.1 embedding of word
        query_embeddings_word = tf.nn.embedding_lookup(self.embedding_query, self.query)

        # use BiLSTM to encode both context and query embeddings rnn to encoding inputs(query)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        bi_outputs_query, bi_state_query = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, query_embeddings_word, dtype=tf.float32, time_major=False, scope="query")
        # query representation. bi_outputs_query contain two elements.one for forward part, another for backward part
        self.rQ = tf.concat([bi_outputs_query[0], bi_outputs_query[1]], axis=-1)
        # [none,hidden_size*2].the concatenation of both directions' last hidden state.
        self.uQ = tf.concat([bi_state_query[0][1], bi_state_query[1][1]], axis=-1)


        context_embeddings_word = tf.nn.embedding_lookup(self.embedding_word, self.context)
        bi_outputs_context, bi_state_context = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, context_embeddings_word, dtype=tf.float32, time_major=False, scope="context")
        # context representation
        self.rP = tf.concat([bi_outputs_context[0], bi_outputs_context[1]], axis=-1)

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
                # return: alignment matrix A between query and context
                self.M_2 = self.query_based_similarity_matching()

                # [none,hidden_size*2] context-based similarity matching
                self.m_3 = self.context_based_similarity_matching()

                # [none,article_len,hidden_size*2] integrated hierarchical matching
                M = self.integrated_hierarchical_matching()

                # [none,article_len,hidden_size*2] add an additional gate
                M = self.add_gate(M)

                # [none,article_len,hidden_size*2] pass through a bi-directional LSTM.
                self.O = self.bi_directional_LSTM(M, "matching")

                # 4.update original passage(context) representation r_p for the use of next layer.
                with tf.variable_scope("dimension_reduction"):
                    # the output of directional LSTM O can be regarded as (refined) context r_p
                    self.r_p = tf.layers.dense(self.O, self.hidden_size * 2)


    def integral_query_matching(self):
        """
        input rP, rQ, uQ
        """
        # [none,1,hidden_size*2]
        u_q_expand = tf.expand_dims(self.uQ, axis=1)
        # [none,article_len] inner product
        inner_product = tf.reduce_sum(tf.multiply(u_q_expand, self.rP), axis=2)
        # [none,article_len] softmax
        c = tf.nn.softmax(inner_product, dim=1)
        # [none,article_len,1] sum of input(context) weighted by attention c
        c_t = tf.expand_dims(c, axis=2)
        # sum of the rP weighted by attention ct
        m_1 = tf.reduce_sum(tf.multiply(c_t, self.rP), axis=1)
        return m_1

    def query_based_similarity_matching(self):

        with tf.variable_scope("query_based_similarity_matching"):
            # [none,article_len,1,hidden_size*2]
            r_p_expand = tf.expand_dims(self.rP, axis=2)
            # [none,1,sentence_len,hidden_size*2]
            r_q_expand = tf.expand_dims(self.rQ, axis=1)
            # [none,article_len,sentence_len]
            r_p_q = tf.multiply(r_p_expand,r_q_expand)
            r_p_tile = tf.tile(r_p_expand, [1, 1, self.sentence_len, 1])
            r_q_tile = tf.tile(r_q_expand, [1, self.context_len, 1, 1])
            # concat
            A = tf.concat([r_p_q, r_p_tile, r_q_tile], axis=-1)
            self.A = tf.squeeze(tf.layers.dense(A, 1), axis=-1)
            # softmax function is performed across the row vector
            B = tf.nn.softmax(self.A, dim=2)
            # [none,article_len,hidden_size*2] each attention vector
            M_2 = tf.matmul(B, self.rQ)
        return M_2

    def context_based_similarity_matching(self):
        # the most representative word
        e = tf.reduce_max(self.A, axis=2)
        # attention
        d = tf.nn.softmax(e, dim=1)
        d_expand = tf.expand_dims(d, axis=-1)
        m_3 = tf.reduce_sum(tf.multiply(d_expand, self.r_p), axis=1)
        return m_3

    def integrated_hierarchical_matching(self):
        # linear function to get the integrated hierarchical matching module
        with tf.variable_scope("integrated_hierarchical_matching"):
            m_1 = tf.expand_dims(self.m_1, axis=1)
            m_3 = tf.expand_dims(self.m_3, axis=1)
            # M1 and M3 are matrixes that are tiled n times by m_1 and m_3
            M_1 = tf.tile(m_1, [1, self.context_len, 1])
            M_3 = tf.tile(m_3, [1, self.context_len, 1])

            M_1 = tf.layers.dense(M_1, self.hidden_size*2)
            M_2 = tf.layers.dense(self.M_2, self.hidden_size*2)
            M_3 = tf.layers.dense(M_3, self.hidden_size * 2)
            # linear function that could be tuned
            M = M_1 + 0.5 * M_2 + 0.5 * M_3
        return M

    def add_gate(self, M):
        # filtrates the part of tokens that are helpful in understanding the relation between passage and query.
        with tf.variable_scope("add_gate"):
            # add bias
            b = 0
            gt = tf.sigmoid(tf.layers.dense(M, 1) + b)
            M_star = tf.multiply(gt, M)
        return M_star

    def bi_directional_LSTM(self,input,scope):
        """
        bi-directional LSTM. input:[none,sequence_length,h]
        :param input:
        :param scope:
        :return: [none,sequence_length,h*2]
        """
        with tf.variable_scope(scope):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, dtype=tf.float32, time_major=False, swap_memory=True, scope=scope)
            result = tf.concat([bi_outputs[0], bi_outputs[1]],axis=-1) # bi_outputs_query contain two elements.one for forward part, another for backward part
        return result #[none,sequence_length,h*2]

    def output_layer(self):
        # initialize the hidden state of the pointer network by a query-aware representation.
        start_point_pred = None
        end_point_pred = None
        with tf.variable_scope("output_layer"):
            l_0 = self.initial_hidden_state()
            # k = 1,2 represent the start point and end point of the answer
            l_k_1 = l_k_2 = l_0
            for i in range(self.output_hop_times):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                start_prediction, end_prediction, v_k_1, v_k_2 = self.point_network(l_k_1, l_k_2)

                # use GRU to update l_k with v_k as input
                l_k_1 = self.GRU(1, l_k_1, v_k_1)
                l_k_2 = self.GRU(2, l_k_2, v_k_2)
        return start_point_pred, end_point_pred

    def initial_hidden_state(self):
        # initialize the hidden state of the pointer network by a query-aware representation
        with tf.variable_scope("initial_hidden_state"):
            z_ = tf.layers.dense(self.rQ, self.hidden_size * 2, activation=tf.nn.tanh, use_bias=True)
            z = tf.layers.dense(z_, 1)
            a = tf.nn.softmax(z, dim=1)

            l_0 = tf.reduce_sum(tf.multiply(a, self.rQ), axis=1)
        return l_0

    def point_network(self, l_k_1, l_k_2):
        """
        point network
        :param O: context representation: [none,article_len,hidden_size*2]. it is a output of matching layer.
        :param l_0: query-aware representation:[none,hidden_size*2]
        :return:
        """
        with tf.variable_scope("point_network"):
            p_start, v_k_1 = self.point_network_single(1, l_k_1)
            p_end, v_k_2 = self.point_network_single(2, l_k_2)
        return p_start, p_end, v_k_1, v_k_2

    def point_network_single(self,k,l_k):
        """
        point network,single
        :param k: k=1,2 represent the start point and end point of the answer
        :param l_k: [none,hidden_size*2].
        :return: p:[none,]
        :return: v_k:[none,hidden_size*4]
        """
        with tf.variable_scope("point_network"+str(k)):
            part_1 = tf.layers.dense(self.O,self.hidden_size * 2) #[none,article_len,hidden_size*2]; where self.O is [none,article_len,hidden_size*2].
            part_2 = tf.expand_dims(tf.layers.dense(l_k,self.hidden_size * 2),axis=1)
            z_ = tf.nn.tanh(part_1+part_2) #[none,article_len,hidden_size*2]
            z = tf.squeeze(tf.layers.dense(z_,1),axis=2) #[none,article_len]
            a = tf.nn.softmax(z,dim=1) #[none,article_len]
            #p=tf.argmax(a,axis=1) #[none,]
            v_k = tf.multiply(tf.expand_dims(a,axis=2),self.O) #[none,article_len,hidden_size*2]
            v_k = tf.reduce_sum(v_k,axis=1) #[none,hidden_size*2]
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
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=self.learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op