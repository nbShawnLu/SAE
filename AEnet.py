import tensorflow as tf
from tensorflow.contrib import layers


class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, re_constant3=1.0,
                 batch_size=200, reg=None, \
                 denoise=False, model_path=None, restore_path=None, \
                 logs_path='./logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        usereg = 2
        # input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])

        weights = self._initialize_weights()

        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)

        # self.Coef = tf.Variable(np.eye(batch_size,batch_size,0,np.float32))
        raws = latent.shape
        rawd = raws[1] * raws[2] * raws[3]
        rawd = rawd.value
        z = tf.reshape(latent, [batch_size, rawd])
        ###
        # zslice1 = tf.slice(z,[0,0],[batch_size,rawd//2])
        # zslice2 = tf.slice(z, [0, rawd//2], [batch_size, rawd // 2])
        # z = zslice1
        ###
        Coef = weights['Coef']
        z_c = tf.matmul(Coef, z)
        self.Coef = Coef
        ###
        # z_c = tf.concat([z_c1,zslice2],axis=1)
        ###
        latent_c = tf.reshape(z_c, tf.shape(latent))
        self.z = z

        self.x_r = self.decoder(latent_c, weights, shape)

        # l_2 reconstruction loss
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))
        tf.summary.scalar("recons_loss", self.reconst_cost)

        if usereg == 2:
            self.reg_losses = tf.reduce_sum(tf.pow(self.Coef, 2.0))
        else:
            self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))

        # reg_constant3 = 1.0
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))
        # XX = tf.matmul(z,tf.transpose(z))
        # ZXXZ = tf.matmul(z_c,tf.transpose(z_c))
        # self.selfexpress_losses2 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(XX, ZXXZ), 2.0))

        # re_constant2 = 1.0
        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        x_flattten = tf.reshape(x_input, [batch_size, -1])
        XZ = tf.matmul(Coef, x_flattten)
        self.selfexpress_losses2 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(XZ, x_flattten), 2.0))

        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses
        self.loss2 = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant3 * self.selfexpress_losses2 + re_constant2 * self.selfexpress_losses

        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss2)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer = self.optimizer2#tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # GradientDescentOptimizer #AdamOptimizer

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
        # [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['Coef'] = tf.Variable(
            0.000001 * tf.random_normal([self.batch_size, self.batch_size], dtype=tf.float32), name='Coef')

        all_weights['enc_w0'] = tf.get_variable("enc_w0",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))  # , name = 'enc_b0'

        iter_i = 1
        while iter_i < n_layers:
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.get_variable(enc_name_wi,
                                                       shape=[self.kernel_size[iter_i], self.kernel_size[iter_i],
                                                              self.n_hidden[iter_i - 1], \
                                                              self.n_hidden[iter_i]],
                                                       initializer=layers.xavier_initializer_conv2d(),
                                                       regularizer=self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.Variable(
                tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))  # , name = enc_name_bi
            iter_i = iter_i + 1

        iter_i = 1
        while iter_i < n_layers:
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers - iter_i],
                                                                           self.kernel_size[n_layers - iter_i],
                                                                           self.n_hidden[n_layers - iter_i - 1],
                                                                           self.n_hidden[n_layers - iter_i]],
                                                       initializer=layers.xavier_initializer_conv2d(),
                                                       regularizer=self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.Variable(
                tf.zeros([self.n_hidden[n_layers - iter_i - 1]], dtype=tf.float32))  # , name = dec_name_bi
            iter_i = iter_i + 1

        dec_name_wi = 'dec_w' + str(iter_i - 1)
        all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                                       self.n_hidden[0]],
                                                   initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        dec_name_bi = 'dec_b' + str(iter_i - 1)
        all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype=tf.float32))  # , name = dec_name_bi

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'),
                                weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())

        n_layers = len(self.n_hidden)
        iter_i = 1
        while iter_i < n_layers:
            layeri = tf.nn.bias_add(
                tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[1, 2, 2, 1], padding='SAME'),
                weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())
            iter_i = iter_i + 1

        layer3 = layeri
        return layer3, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        n_layers = len(self.n_hidden)
        layer3 = z
        iter_i = 0
        while iter_i < n_layers:
            # if iter_i == n_layers-1:
            #    strides_i = [1,2,2,1]
            # else:
            #    strides_i = [1,1,1,1]
            shape_de = shapes[n_layers - iter_i - 1]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack(
                [tf.shape(self.x)[0], shape_de[1], shape_de[2], shape_de[3]]), strides=[1, 2, 2, 1], padding='SAME'),
                            weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
            iter_i = iter_i + 1
        return layer3

    def partial_fit(self, X, lr, mode=0):  #
        if mode == 'fine':
            cost0, cost1, cost2, summary, _, Coef = self.sess.run((self.reconst_cost, self.selfexpress_losses,
                                                                   self.selfexpress_losses2, self.merged_summary_op,
                                                                   self.optimizer2, self.Coef),
                                                                  feed_dict={self.x: X, self.learning_rate: lr})  #
        else:
            cost0, cost1, cost2, summary, _, Coef = self.sess.run((self.reconst_cost, self.selfexpress_losses,
                                                                   self.selfexpress_losses2, self.merged_summary_op,
                                                                   self.optimizer, self.Coef),
                                                                  feed_dict={self.x: X, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return [cost0, cost1, cost2], Coef

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")