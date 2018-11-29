from QANet.params import Params as param
import QANet.util as util
import tensorflow as tf
import numpy as np


# a tensorflow computation graph is treated as an object of the Graph class
class Graph(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # paceholders for inputs and outputs
            B, N, M, C = param.batch_size, param.max_context_words, param.max_question_words, param.max_chars
            # get inputs and outputs
            self.x_c_w, self.x_c_c, self.x_q_w, self.x_q_c, self.y = my.get_batch_data()
            '''
            #can also use placeholders as below if needed:
            #input sequence of word vocabulary indices of the context
            self.x_c_w = tf.placeholder(tf.int32, shape=[B, N], name="context_words")
            #input sequence of char vocabulary indices (0 to 25) of the words of the context
            self.x_c_c = tf.placeholder(tf.int32, shape=[B, N, C], name="context_word_chars")
            #input sequence of question vocabulary indices of the context
            self.x_q_w =  tf.placeholder(tf.int32, shape=[B, M], name="question_words")
            #input sequence of char vocabulary indices (0 to 25) of the words of the question
            self.x_q_c = tf.placeholder(tf.int32, shape=[B, M, C], name="context_question_chars")
            #output as a one hot encoding of the start position and end position indices over the context words
            self.y = tf.placeholder(tf.int32, shape=[B, N, 2], name="out")
            '''

            '''          
            part1: an embedding layer
            '''
            VW, VC, DW, DC = param.word_vocab_size, param.char_vocab_size, param.word_emb_dim, param.char_emb_dim
            # compute word embeddings of the context words through 300 dimensional GloVe embedding
            self.x_c_w_emb = util.embedding(inputs=self.x_c_w, shape=[VW, DW], scope="word_embedding", reuse=None)
            # compute word embeddings of the question words through 300 dimensional GloVe embedding
            self.x_q_w_emb = util.embedding(inputs=self.x_q_w, scope="word_embedding", reuse=True)
            # compute through character embeddings of the context words
            self.x_c_c_emb = util.embedding(inputs=self.x_c_c, shape=[VC, DC], scope="char_embedding", reuse=None)
            # compute character embeddings of the question words
            self.x_q_c_emb = util.embedding(inputs=self.x_q_c, scope="char_embedding", reuse=True)

            # max pooling over character embeddings to get fixed size embedding of each word
            self.x_c_c_emb = tf.reduce_max(self.x_c_c_emb, reduction_indices=[2])
            # concatenate GloVe embedding with character embedding
            self.x_c_emb = tf.concat(values=[self.x_c_w_emb, self.x_c_c_emb], axis=2, name="x_context_emb")
            # max pooling over character embeddings to get fixed size embedding of each word
            self.x_q_c_emb = tf.reduce_max(self.x_q_c_emb, reduction_indices=[2])
            # concatenate GloVe embedding with character embedding
            self.x_q_emb = tf.concat(values=[self.x_q_w_emb, self.x_q_c_emb], axis=2, name="x_question_emb")

            # apply a highway network of 2 layers on top of computed embedding
            self.x_c_emb = util.highway_network(inputs=self.x_c_emb, num_layers=param.highway_num_layers, use_bias=True,
                                              transform_bias=-1.0, scope='highway_net', reuse=None)
            self.x_q_emb = util.highway_network(inputs=self.x_q_emb, num_layers=param.highway_num_layers, use_bias=True,
                                              transform_bias=-1.0, scope='highway_net', reuse=True)

            '''
            part2: an embedding encoder layer
            '''
            # single encoder block: convolution_layer X # + self_attention_layer + feed_forward_layer
            # apply 1 encoder stack of 1 encoder block on context embedding
            self.x_c_enc = util.encoder_block(inputs=self.x_c_emb, num_conv_layer=4, filters=128, kernel_size=7,
                                            num_att_head=8, scope='encoder_block', reuse=None)
            # apply 1 encoder stack of 1 encoder block on question embedding
            self.x_q_enc = util.encoder_block(inputs=self.x_q_emb, num_conv_layer=4, filters=128, kernel_size=7,
                                            num_att_head=8, scope='encoder_block', reuse=True)

            '''           
            part3: a context-query attention layer
            '''
            # apply a context-query attention layer to compute context-to-query attention and query-to-context attention
            self.att_a, self.att_b = util.context_query_attention(context=self.x_c_enc, query=self.x_q_enc,
                                                                scope='context_query_att', reuse=None)

            '''
            part4: a model encoder layer
            '''
            # apply 3 encoder stacks of 7 encoder blocks            
            # prepare input as [c, a, c dot a, c dot b] where a and b are rows of attention matrix A (att_a) and B (att_b)
            # computing c dot a
            self.c_mult_att_a = tf.multiply(self.x_c_enc, self.att_a)
            # computing c dot b
            self.c_mult_att_b = tf.multiply(self.x_c_enc, self.att_b)
            # computing [c, a, c dot a, c dot b] 
            # NOTE: there is an ambiguity here. Since the encoder blocks have to share weights, the input dimensions to each block should remain same, however the startig input is mentioned as a concatenation of four 128 dimensional (=512) hidden states [c, a, c dot a, c dot b] while the blocks above the first block will have inputs of 128 dimensional since a 1D convolution will map the first 512 dimensional input to a 128 dimensional output. To overcome this, an average composition instead of a concat is used over (c, a, c dot a, c dot b)
            # compute average of [c, a, c dot a, c dot b] tensors 
            # dimension=[B, N, d] ([batch_size, max_words_context, hidden_dimension=128])
            self.model_enc = tf.reduce_mean(tf.concat(
                [tf.expand_dims(self.x_c_enc, 2), tf.expand_dims(self.att_a, 2), tf.expand_dims(self.c_mult_att_a, 2),
                 tf.expand_dims(self.c_mult_att_b, 2)], axis=2), axis=2, name="model_enc_inp")
            # for each encoder stack
            for i in range(3):
                # for each encoder block within each stack           
                for j in range(7):
                    # the call to the first model encoder block in each stack will have reuse None to create new weight tensors
                    if (i == 0):
                        self.model_enc = util.encoder_block(inputs=self.model_enc, num_conv_layer=2, filters=128,
                                                          kernel_size=5, num_att_head=8,
                                                          scope='model_enc_block_{}'.format(j), reuse=None)
                    # subsequent blocks in each stack (block 2 to 7) will have reuse True since each stack shares weights across blocks
                    else:
                        self.model_enc = util.encoder_block(inputs=self.model_enc, num_conv_layer=2, filters=128,
                                                          kernel_size=5, num_att_head=8,
                                                          scope='model_enc_block_{}'.format(j), reuse=True)
                # after completion of first encoder stack, store output as M0
                if (i == 1):
                    # store model_enc as output M0 after completion of run of first stack of model encoder blocks
                    # model encoder blocks executed: 7
                    # using tf.identity to copy a tensor
                    self.out_m0 = tf.identity(self.model_enc)
                    # store model_enc as output M1 after completion of run of second stack of model encoder blocks
                    # model encoder blocks executed: 14
                # after completion of second encoder stack, store output as M1
                elif (i == 2):
                    self.out_m1 = tf.identity(self.model_enc)
                    # store model_enc as output M2 after completion of run of third stack of model encoder blocks
                    # model encoder blocks executed: 21
                # after completion of third encoder stack, store output as M2
                else:
                    self.out_m2 = tf.identity(self.model_enc)

            '''        
            part5: an output layer      
            '''
            # feature vector for position 1 is [M0;M1]
            self.inp_pos1 = tf.concat((self.out_m0, self.out_m1), axis=2)
            # feature vector for position 2 is [M0;M2]
            self.inp_pos2 = tf.concat((self.out_m0, self.out_m2), axis=2)
            # compute softmax probability scores on positions of context words for being position 1
            self.pos1 = tf.nn.softmax(tf.layers.dense(self.inp_pos1, 1, activation=tf.tanh, name='dense_pos1'))
            # compute softmax probability scores on positions of context words for being position 2
            self.pos2 = tf.nn.softmax(tf.layers.dense(self.inp_pos2, 1, activation=tf.tanh, name='dense_pos2'))
            # concatenate both prediction vectors
            # dimensions=[B, N, 2] ([batch_size, max_context_words, 2])
            self.pred = tf.concat((self.pos1, self.pos2), axis=-1)
            # loss = -mean(log(p1) + log(p2)) = mean(-log(p1*p2))
            self.loss = tf.reduce_mean(
                -tf.log(tf.reduce_prod(tf.reduce_sum(self.pred * tf.cast(self.y, 'float'), 1), 1) + param.epsilon_1))

            # training scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # using ADAM optimizer with beta1=0.8, beta2=0.999 and epsilon=1e-7
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr, beta1=param.beta1, beta2=param.beta2,
                                                    epsilon=param.epsilon_2)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            # loss summary
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()