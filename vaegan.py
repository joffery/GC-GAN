import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images
from utils import CelebA, getImageData, celeba_label
from tensorflow.contrib.layers.python.layers import layer_norm
import numpy as np
# import cv2
import os
import random
from vggface import vgg_face
import pickle

TINY = 1e-8

class vaegan(object):

    #build model
    def __init__(self, batch_size, max_epoch, model_path, train_data, test_data, latent_dim, sample_path, exp_transfer_path, lm_interpolation_path, log_dir, learnrate_init):

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.saved_model_path = model_path
        self.ds_train = train_data
        self.ds_test = test_data
        self.latent_dim = latent_dim
        self.sample_path = sample_path
        self.exp_transfer_path = exp_transfer_path
        self.lm_interpolation_path = lm_interpolation_path

        self.log_dir = log_dir
        self.learn_rate_init = learnrate_init
        self.log_vars = []
        self.test_lm_transfer_path = 'test_lm_transfer'
        self.channel = 3
        self.output_size = CelebA().image_size
        self.local_size = int(self.output_size/2)

        #总觉得在这里用分类不太好
        self.lambda_cls = 0.0001
        self.lambda_embed = 0.0001
        self.lambda_gen = 0.001
        self.landmark_size = 68*2
        self.landmark_embed_size = 32
        # self.tv_penalty = 0.0001
        self.lambda_kl = 0.001

        #输入的NE图像
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        #和landmark对应的表情图像, 输入有landmark点, 输出没有
        self.images_emotion = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.real_local_imgs= tf.placeholder(tf.float32, [self.batch_size, self.local_size, self.local_size, self.channel])
        self.fake_local_imgs= tf.placeholder(tf.float32, [self.batch_size, self.local_size, self.local_size, self.channel])

        #输入图像的landmark通道
        self.images_lm = tf.placeholder(tf.float32, [self.batch_size, self.landmark_size])
        self.emotion_images_lm = tf.placeholder(tf.float32, [self.batch_size, self.landmark_size])
        self.emotion_images_lm_reference = tf.placeholder(tf.float32, [self.batch_size, self.landmark_size])
        self.emotion_label = tf.placeholder(tf.int32, [self.batch_size])
        self.emotion_label_reference = tf.placeholder(tf.int32, [self.batch_size])
        self.lm_embed_input = tf.placeholder(tf.float32, [self.batch_size, self.landmark_embed_size])
        self.isTrain = tf.placeholder(tf.bool)

        #z
        self.z_p = tf.placeholder(tf.float32, [self.batch_size , self.latent_dim])
        # the noise is fixed, not changed, 可能是存在一定问题的
        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim], seed=1000)

        # 定义好bn

    def build_model_vaegan(self):

        #vae版本
        # self.z_mean, self.z_sigm, self.conv1, self.conv2, self.conv3 = self.Encode(self.images)
        # self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)

        self.z_x, self.conv1, self.conv2, self.conv3 = self.Encode_AE(self.images)

        # landmark
        self.lm_embed, self.lm_recon = self.Embed_landmark(self.emotion_images_lm)
        # metric learning, 用于更新embed网络, 使其学到用户无关的特征, disentangled
        self.lm_embed_reference, _ = self.Embed_landmark(self.emotion_images_lm_reference, reuse=True)

        self.x_tilde = self.image_with_landmark_embed(self.z_x, self.lm_embed, reuse=False)
        self.x_tilde_input = self.image_with_landmark_embed(self.z_x, self.lm_embed_input, reuse=True)

        # 重建损失
        self.recon_loss = self.NLLNormal(self.x_tilde, self.images_emotion)

        # identity loss
        # self.input_vgg_feature = vgg_face('vgg-face.mat', self.images)
        # self.x_tilde_vgg_feature = vgg_face('vgg-face.mat', self.x_tilde)
        # self.identity_loss = self.NLLNormal(self.input_vgg_feature, self.x_tilde_vgg_feature)

        # tv loss
        # self.tv_loss = (tf.nn.l2_loss(self.x_tilde[:, 1:, :, :] - self.x_tilde[:, :self.output_size - 1, :, :]) / self.output_size
        #            + tf.nn.l2_loss(self.x_tilde[:, :, 1:, :] - self.x_tilde[:, :, :self.output_size - 1, :]) / self.output_size)

        # global D
        # 重建出来的样本经过D网络
        self.fake_mid_out, self.fake_out, self.fake_emo_aux = self.discriminate(self.x_tilde)
        # 真实图像经过D网络
        self.real_mid_out, self.real_out, self.real_emo_aux = self.discriminate(self.images_emotion, True)

        # 分类损失
        self.real_emotion_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.emotion_label, logits=self.real_emo_aux))
        self.fake_emotion_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.emotion_label, logits=self.fake_emo_aux))
        # WGAN-GP
        wd = tf.reduce_mean(tf.reduce_sum(self.real_out,1)) - tf.reduce_mean(tf.reduce_sum(self.fake_out,1))
        gp = self.gradient_penalty(self.images_emotion, self.x_tilde)
        self.D_loss_global = -wd + gp * 10.0
        self.G_tilde_loss_global = -tf.reduce_mean(tf.reduce_sum(self.fake_out,1))

        # # local D
        # # 重建出来的样本经过D网络
        # self.fake_out_local = self.discriminate_local(self.fake_local_imgs)
        # # 真实图像经过D网络
        # self.real_out_local = self.discriminate_local(self.real_local_imgs, True)
        # # WGAN-GP
        # wd_local = tf.reduce_mean(self.real_out_local) - tf.reduce_mean(self.fake_out_local)
        # gp_local = self.gradient_penalty(self.real_local_imgs, self.fake_local_imgs, local=True)
        # self.D_loss_local = -wd_local + gp_local * 10.0
        # self.G_tilde_loss_local = -tf.reduce_mean(self.fake_out_local)

        # Kl loss
        # self.kl_loss = self.KL_loss()

        # preceptual loss
        self.LL_loss = self.NLLNormal(self.fake_mid_out, self.real_mid_out)

        # for D
        self.D_loss = self.D_loss_global + self.lambda_cls*self.real_emotion_cls_loss

        # for E
        self.encode_loss = self.recon_loss + self.lambda_kl*self.LL_loss\
                           + self.lambda_cls*self.fake_emotion_cls_loss

        #for G
        self.G_loss = self.lambda_gen*(self.G_tilde_loss_global)\
                      + self.recon_loss + self.lambda_cls*self.fake_emotion_cls_loss

        #embed network loss
        # self.Embed_loss
        self.embed_loss()

        self.log_info()

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis_' in var.name]
        self.embed_vars = [var for var in t_vars if 'em_' in var.name]
        self.g_vars = [var for var in t_vars if 'gen_' in var.name] + self.embed_vars
        self.e_vars = [var for var in t_vars if 'e_' in var.name]

        # print('log', self.log_vars)
        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def image_with_landmark_embed(self, z_x, lm_embed, reuse=False):
        z_x_lm = tf.concat([z_x, lm_embed], axis=-1)
        #重建出来的样本, 是4个通道的,最后一个通道是mask
        x_tilde_mix = self.generate(z_x_lm, self.conv1, self.conv2, self.conv3, reuse = reuse)
        return x_tilde_mix

    #do train
    def train(self):
        global_step = tf.Variable(0, trainable=False)
        add_global = global_step.assign_add(1)
        new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=global_step, decay_steps=20000,
                                                   decay_rate=0.98)
        #for D
        trainer_D = tf.train.AdamOptimizer(learning_rate=new_learning_rate, beta1=0.5)
        gradients_D = trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
        # clipped_gradients_D = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients_D]
        opti_D = trainer_D.apply_gradients(gradients_D)

        #for G
        trainer_G = tf.train.AdamOptimizer(learning_rate=new_learning_rate, beta1=0.5)
        gradients_G = trainer_G.compute_gradients(self.G_loss, var_list=self.g_vars)
        # clipped_gradients_G = [(tf.clip_by_value(_[0], -1, 1.), _[1]) for _ in gradients_G]
        opti_G = trainer_G.apply_gradients(gradients_G)

        #for E
        trainer_E = tf.train.AdamOptimizer(learning_rate=new_learning_rate, beta1=0.5)
        gradients_E = trainer_E.compute_gradients(self.encode_loss, var_list=self.e_vars)
        # clipped_gradients_E = [(tf.clip_by_value(_[0], -1, 1.), _[1]) for _ in gradients_E]
        opti_E = trainer_E.apply_gradients(gradients_E)

        #for Embed
        trainer_Embed = tf.train.AdamOptimizer(learning_rate=new_learning_rate, beta1=0.5)
        gradients_Embed = trainer_Embed.compute_gradients(self.Embed_loss, var_list=self.embed_vars)
        # clipped_gradients_E = [(tf.clip_by_value(_[0], -1, 1.), _[1]) for _ in gradients_E]
        opti_Embed = trainer_Embed.apply_gradients(gradients_Embed)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            embed_vector = []
            embed_label = []

            sess.run(init)
            # 从断点处继续训练
            self.saver.restore(sess, self.saved_model_path)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            batch_num = 0
            e = 0
            step = 0
            counter = 0
            # 训练之前就应该shuffle一次
            self.ds_train = self.shuffle_train(self.ds_train)
            while e <= self.max_epoch:

                max_iter = len(self.ds_train)/self.batch_size - 1
                while batch_num < max_iter:

                    step = step + 1

                    if batch_num >= max_iter - 1:
                        self.ds_train = self.shuffle_train(self.ds_train)

                    train_list = CelebA.getNextBatch(self.ds_train, batch_num, self.batch_size)

                    ne_imgs,  emo_imgs, ne_lm, emo_lm, emo_label, emo_lm_ref, emo_label_ref\
                        = CelebA.getTrainImages_lm_embed(train_list)

                    sample_z = np.random.normal(size=[self.batch_size, self.latent_dim])

                    # WGAN-GP
                    loops = 5

                    # # 先获取到当前G网络生成的图像
                    # fake_emo_imgs = sess.run(self.x_tilde, feed_dict={self.images: ne_imgs,
                    #                                                   self.images_lm: ne_lm,
                    #                                                   self.emotion_images_lm: emo_lm,
                    #                                                   self.isTrain: True})
                    # # optimization D
                    # local_x_batch, local_completion_batch = self.crop_local_imgs(emo_imgs, fake_emo_imgs)

                    # T-SNE
                    # embed_v = sess.run(self.lm_embed,  feed_dict={self.emotion_images_lm: emo_lm, self.isTrain:False} )
                    # embed_vector.append(embed_v)
                    # embed_label.append(emo_label)
                    #
                    # if step == 20:
                    #     with open('embed_vector.pickle', 'wb') as f:
                    #         pickle.dump(embed_vector, f, protocol=-1)
                    #     with open('embed_label.pickle', 'wb') as f:
                    #         pickle.dump(embed_label, f, protocol=-1)

                    for _ in range(loops):
                        sess.run(opti_D, feed_dict={self.images: ne_imgs, self.z_p: sample_z, self.images_emotion: emo_imgs,
                                                    # self.real_local_imgs: local_x_batch, self.fake_local_imgs: local_completion_batch,\
                                                    self.images_lm: ne_lm, self.emotion_label: emo_label, self.emotion_images_lm: emo_lm, self.isTrain: True})

                    # 后面再改 self.images_lm
                    for _ in range(1):

                        #optimization Embed
                        sess.run(opti_Embed, feed_dict={self.emotion_label: emo_label, self.emotion_images_lm: emo_lm, self.isTrain: True,
                                                        self.emotion_label_reference: emo_label_ref, self.emotion_images_lm_reference: emo_lm_ref})

                        #optimization E
                        sess.run(opti_E, feed_dict={self.images: ne_imgs, self.images_emotion: emo_imgs, self.images_lm: ne_lm, self.isTrain: True,
                                                    # self.real_local_imgs: local_x_batch, self.fake_local_imgs: local_completion_batch,
                                                    self.emotion_label: emo_label, self.emotion_images_lm: emo_lm})
                        #optimizaiton G
                        sess.run(opti_G, feed_dict={self.images: ne_imgs, self.z_p: sample_z, self.images_emotion: emo_imgs, self.images_lm: ne_lm, self.isTrain: True,
                                                    # self.real_local_imgs: local_x_batch, self.fake_local_imgs: local_completion_batch,
                                                    self.emotion_label: emo_label, self.emotion_images_lm: emo_lm})

                    summary_str = sess.run(summary_op, feed_dict = {self.images:ne_imgs, self.z_p: sample_z, self.images_emotion: emo_imgs, self.images_lm: ne_lm,
                                                                    self.emotion_label: emo_label, self.emotion_images_lm: emo_lm,
                                                                    self.emotion_label_reference: emo_label_ref, self.isTrain: False,
                                                                    # self.real_local_imgs: local_x_batch, self.fake_local_imgs: local_completion_batch,
                                                                    self.emotion_images_lm_reference: emo_lm_ref})
                    summary_writer.add_summary(summary_str , step)

                    batch_num += 1

                    new_learn_rate = sess.run(new_learning_rate)
                    if new_learn_rate > 0.00005:
                        sess.run(add_global)

                    if step%20 == 0:
                        D_loss, fake_loss, encode_loss, LL_loss, kl_loss, recon_loss, positive_loss, negtive_loss, lm_recon_loss, Embed_loss, real_cls, fake_cls = sess.run(
                            [self.D_loss, self.G_loss, self.encode_loss, self.D_loss, self.LL_loss,
                             self.recon_loss, self.positive_loss, self.negative_loss, self.lm_recon_loss, self.Embed_loss, self.real_emotion_cls_loss, self.fake_emotion_cls_loss],
                            feed_dict={self.images: ne_imgs, self.z_p: sample_z,
                                       self.images_emotion: emo_imgs, self.images_lm: ne_lm,
                                       self.emotion_label: emo_label, self.emotion_images_lm: emo_lm, self.isTrain: False,
                                       # self.real_local_imgs: local_x_batch, self.fake_local_imgs: local_completion_batch,
                                       self.emotion_label_reference: emo_label_ref, self.emotion_images_lm_reference: emo_lm_ref})
                        print(
                            "EPOCH %d step %d: D: loss = %.7f G: loss=%.7f Encode: loss=%.7f identity loss=%.7f KL=%.7f recon_loss=%.7f "
                            "positive_loss=%.7f negtive_loss=%.7f lm_recon_loss=%.7f Embed_loss==%.7f real_cls=%.7f fake_cls=%.7f" % (
                            e, step, D_loss, fake_loss, encode_loss, LL_loss, kl_loss, recon_loss,
                            positive_loss, negtive_loss, lm_recon_loss, Embed_loss,real_cls, fake_cls))
                    # previous
                    if np.mod(step , 20) == 1:
                        self.ds_test = self.shuffle_train(self.ds_test)
                        test_list = CelebA.getNextBatch(self.ds_test, 0, self.batch_size)

                        self.test_basic(sess, test_list, 0, step)
                        self.test_landmark_interpolation(sess, test_list, 0, step)
                        self.test_expression_transfer(sess, 0, step)
                        self.saver.save(sess , self.saved_model_path)

                    # for tsne interpolation
                    # if step > 0:
                    #     print('step', step)
                    #     self.ds_test = self.shuffle_train(self.ds_test)
                    #     test_list = CelebA.getNextBatch(self.ds_test, 0, self.batch_size)
                    #
                    #     self.test_basic(sess, test_list, 0, step)
                    #     embed_inter_v, embed_img_n = self.test_landmark_interpolation(sess, test_list, 0, step)
                    #     embed_inter_vector.append(embed_inter_v)
                    #     embed_img_names.append(embed_img_n)
                    #
                    #     if step == 20:
                    #         with open('embed_inter_vector.pickle', 'wb') as f:
                    #             pickle.dump(embed_inter_vector, f, protocol=-1)
                    #         with open('embed_img_names.pickle', 'wb') as f:
                    #             pickle.dump(embed_img_names, f, protocol=-1)
                    #
                    #     # self.test_one_eye_close(sess, test_list, 0, step)
                    #     # self.test_expression_transfer(sess, test_list, 0, step)
                    #     self.saver.save(sess, self.saved_model_path)




                e += 1
                batch_num = 0
            save_path = self.saver.save(sess , self.saved_model_path)
            print ("Model saved in file: %s" % save_path)

    def test_basic(self, sess, test_list, e, step):

        ne_imgs, emo_imgs, ne_lm, emo_lm, emo_label, \
        emo_lm_ref, emo_label_ref = CelebA.getTrainImages_lm_embed(test_list)

        # input image
        save_images(ne_imgs[0:64], [8, 8],
                    '{}/train_{:02d}_{:04d}_in.png'.format(self.sample_path, e, step))

        # ground truth
        save_images(emo_imgs[0:64], [8, 8],
                    '{}/train_{:02d}_{:04d}_r.png'.format(self.sample_path, e, step))

        # generate image
        sample_images = sess.run(self.x_tilde, feed_dict={self.images: ne_imgs, self.images_lm: ne_lm, self.isTrain: False,
                                                          self.emotion_images_lm: emo_lm})
        save_images(sample_images[0:64], [8, 8], '{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e, step))

    def test_expression_transfer(self, sess, e, step):

        realbatch_array_test, realbatch_array_emtoions_lm = CelebA.landmark_transfer_test()

        print()
        # print('realbatch_array', realbatch_array_test.shape)
        batch_size = self.batch_size
        img_width = int(np.sqrt(batch_size))
        # input img
        save_images(realbatch_array_test[0:batch_size], [img_width, img_width],
                    '{}/train_{:02d}_{:04d}_in.png'.format(self.exp_transfer_path, e, step))

        # 生成的图像
        sample_images = sess.run(self.x_tilde, feed_dict={self.images: realbatch_array_test, self.isTrain:False,
                                                          self.emotion_images_lm: realbatch_array_emtoions_lm})
        save_images(sample_images[0:batch_size], [img_width, img_width],
                    '{}/train_{:02d}_{:04d}.png'.format(self.exp_transfer_path, e, step))

    def test_landmark_interpolation(self, sess, test_list, e, step):

        ne_imgs, emo_imgs, ne_lm, emo_lm, emo_label, \
        emo_lm_ref, emo_label_ref = CelebA.getTrainImages_lm_embed(test_list)

        #可以先测试landmark空间的插值，然后再试试隐空间的插值，以每个batch的第一张图像为基准
        # print('realbatch_array', realbatch_array_test.shape)
        batch_size = self.batch_size
        img_width = int(np.sqrt(batch_size))
        test_img = ne_imgs[0]
        test_img_lm = ne_lm[0]
        test_img_emotion_lm = emo_lm[0]
        test_lm_interpolation = []
        for i in range(64):
            factor = i/63
            test_lm_interpolation.append(factor*test_img_lm+(1-factor)*test_img_emotion_lm)

        test_lm_interpolation = np.asarray(test_lm_interpolation, dtype=np.float32)
        test_imgs = np.repeat(np.expand_dims(test_img, axis=0), 64, axis=0)

        # input img
        save_images(test_imgs[0:batch_size], [img_width, img_width],
                    '{}/train_{:02d}_{:04d}_in.png'.format(self.lm_interpolation_path, 0, step))

        # generate img
        sample_images = sess.run(self.x_tilde, feed_dict={self.images: test_imgs, self.isTrain: False,
                                                          self.emotion_images_lm: test_lm_interpolation})

        save_images(sample_images[0:batch_size], [img_width, img_width],
                    '{}/train_{:02d}_{:04d}.png'.format(self.lm_interpolation_path, 0, step))

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()
            conv1 = tf.nn.relu(conv2d(x_var, output_dim=32, name='dis_conv1'))
            conv2= tf.nn.relu(conv2d(conv1, output_dim=128, name='dis_conv2'))
            conv3= tf.nn.relu(conv2d(conv2, output_dim=256, name='dis_conv3'))
            conv4 = conv2d(conv3, output_dim=256, name='dis_conv4')
            middle_conv = conv4
            conv4 = tf.nn.relu(conv4)
            conv5 = conv2d(conv4, output_dim=1, name='dis_conv5')
            conv5 = tf.reshape(conv5, [self.batch_size, -1])
            print('conv5_shape', conv5.get_shape())

            conv4 = tf.reshape(conv4, [self.batch_size, -1])
            output_aux = fully_connect(conv4, output_size=6, scope='dis_fully1') #6个表情类别

            return middle_conv, conv5, output_aux

    def discriminate_local(self, x_patch, reuse=False):
        with tf.variable_scope("discriminator_local") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = lrelu(conv2d(x_patch, output_dim=64, name='dis_conv1_patch'), 0.01)
            conv2 = lrelu(conv2d(conv1, output_dim=128, name='dis_conv2_patch'))
            conv3 = lrelu(conv2d(conv2, output_dim=256, name='dis_conv3_patch'))
            conv4 = lrelu(conv2d(conv3, output_dim=512, name='dis_conv4_patch'))
            conv4 = lrelu(conv4, [self.batch_size, -1])
            fc1 = tf.nn.relu(fully_connect(conv4, output_size=512, scope='dis_fully1_patch'))
            output = fully_connect(fc1, output_size=1, scope='dis_output_patch')  # 6个表情类别
            return output

    def generate(self, z_var, conv1, conv2, conv3, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse == True:
                scope.reuse_variables()

            d1 = tf.nn.relu(batch_normal(fully_connect(z_var , output_size=4*4*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse, isTrain=self.isTrain))
            d2 = tf.reshape(d1, [self.batch_size, 4, 4, 256])
            d2 = tf.nn.relu(batch_normal(de_conv(d2 , output_shape=[self.batch_size, 8, 8, 256], name='gen_deconv2'), scope='gen_bn2', reuse=reuse, isTrain=self.isTrain))+conv3
            print('d2_shape', d2.get_shape())
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 16, 16, 128], name='gen_deconv3'), scope='gen_bn3', reuse=reuse, isTrain=self.isTrain))+conv2
            print('d3_shape', d3.get_shape())
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 32, 32, 64], name='gen_deconv4'), scope='gen_bn4', reuse=reuse, isTrain=self.isTrain))+conv1
            print('d4_shape()', d4.get_shape())
            d5 = tf.nn.relu(batch_normal(de_conv(d4, output_shape=[self.batch_size, 64, 64, 64], name='gen_deconv5'), scope='gen_bn5', reuse=reuse, isTrain=self.isTrain))
            print('d5_shape', d5.get_shape())
            d6 = conv2d(d5, output_dim=3, d_h=1, d_w=1, name='gen_conv6')
            print('d6_shape', d6.get_shape())
            return tf.nn.tanh(d6)

    def Encode(self, img, reuse=False):

        with tf.variable_scope('encode') as scope:
            if reuse == True:
                scope.reuse_variables()
            conv1 = tf.nn.relu(batch_normal(conv2d(img, output_dim=64, name='e_c1'), scope='e_bn1', reuse=reuse, isTrain=self.isTrain))
            print('conv1_shape', conv1.get_shape())
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2', reuse=reuse, isTrain=self.isTrain))
            print('conv2_shape', conv2.get_shape())
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=256, name='e_c3'), scope='e_bn3', reuse=reuse, isTrain=self.isTrain))
            print('conv3_shape', conv3.get_shape())
            conv3_before_fc = conv3
            conv3 = tf.reshape(conv3, [self.batch_size, 256 * 8 * 8])
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4', reuse=reuse, isTrain=self.isTrain))
            z_mean = fully_connect(fc1, output_size=128, scope='e_f2')
            z_sigma = fully_connect(fc1, output_size=128, scope='e_f3')
            return z_mean, z_sigma, conv1, conv2, conv3_before_fc  #应该是激活之前的值，还是激活之后的值呢？

    def Encode_AE(self, img, reuse=False):

        with tf.variable_scope('encode') as scope:
            if reuse == True:
                scope.reuse_variables()
            conv1 = tf.nn.relu(batch_normal(conv2d(img, output_dim=64, name='e_c1'), scope='e_bn1', reuse=reuse, isTrain=self.isTrain))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2', reuse=reuse, isTrain=self.isTrain))
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=256, name='e_c3'), scope='e_bn3', reuse=reuse, isTrain=self.isTrain))
            conv3_before_fc = conv3
            conv3 = tf.reshape(conv3, [self.batch_size, 256 * 8 * 8])
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4', reuse=reuse, isTrain=self.isTrain))
            z_x = tf.nn.relu(batch_normal(fully_connect(fc1, output_size=128, scope='e_f2'), scope='e_bn5', reuse=reuse, isTrain=self.isTrain))
            return z_x, conv1, conv2, conv3_before_fc

    def Embed_landmark(self, lm, reuse=False):

        with tf.variable_scope('embed') as scope:
            if reuse == True:
                scope.reuse_variables()
            fc1 = tf.nn.relu(batch_normal(fully_connect(lm, output_size=128, scope='em_f1'), scope='em_bn1', reuse=reuse, isTrain=self.isTrain))
            fc2 = tf.nn.relu(batch_normal(fully_connect(fc1, output_size=64, scope='em_f2'), scope='em_bn2', reuse=reuse, isTrain=self.isTrain))
            fc3 = tf.nn.relu(batch_normal(fully_connect(fc2, output_size=32, scope='em_f3'), scope='em_bn3', reuse=reuse, isTrain=self.isTrain))
            fc4 = tf.nn.relu(batch_normal(fully_connect(fc3, output_size=64, scope='em_f4'), scope='em_bn4', reuse=reuse, isTrain=self.isTrain))
            fc5 = tf.nn.relu(batch_normal(fully_connect(fc4, output_size=128, scope='em_f5'), scope='em_bn5', reuse=reuse, isTrain=self.isTrain))
            fc6 = tf.nn.tanh((fully_connect(fc5, output_size=68*2, scope='em_f6')))
            return fc3, fc6

    # 还是不用他这个kl散度了
    def KL_loss(self):
        return tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(self.z_sigm) + self.z_mean ** 2 - 1. - self.z_sigm, 1))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    # 对perceptual loss做了些改变
    def NLLNormal(self, pred, target):
        tmp = tf.reduce_mean(
            tf.reduce_sum(tf.reshape(tf.abs(pred - target), [self.batch_size, -1]), 1))
        return tmp

    def shuffle_train(self, train):
        length = len(train)
        perm = np.arange(length)
        np.random.shuffle(perm)
        input_list = train[perm]
        return input_list

    def gradient_penalty(self, real, fake, local = False):

        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        print('x',x.get_shape())

        if local == False:
            _, pred, _ = self.discriminate(x, reuse=True)
        else:
            pred = self.discriminate_local(x, reuse=True)

        pred = tf.reduce_sum(pred, axis=1, keep_dims=True)
        print('pred', pred.get_shape())
        gradients = tf.gradients(pred, x)[0]
        print('gradient', gradients.get_shape())
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), 1))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    def get_points(self):
        points = []
        for i in range(self.batch_size):
            x1, y1 = np.random.randint(0, self.output_size - self.local_size + 1, 2)
            x2, y2 = np.array([x1, y1]) + self.local_size
            points.append([x1, y1, x2, y2])
        return np.array(points)

    def embed_loss(self):
        self.same_label = tf.cast(tf.equal(self.emotion_label, self.emotion_label_reference), dtype=tf.float32)
        self.diff_label = tf.ones_like(self.same_label) - self.same_label
        self.l2_loss_pairs = tf.reduce_sum(tf.square(self.lm_embed - self.lm_embed_reference), 1)
        self.positive_loss = 0.5 * tf.reduce_mean(self.same_label * self.l2_loss_pairs)
        # 可以调整margin 1确实太近了,不太好
        self.negative_loss = 0.5 * tf.reduce_mean(self.diff_label * tf.nn.relu(5 - self.l2_loss_pairs))
        self.lm_recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.lm_recon - self.emotion_images_lm), 1))

        # self.Embed_loss = self.lambda_embed * (self.positive_loss + self.negative_loss + self.lm_recon_loss)

        # w/o contrastive learning
        self.Embed_loss = self.lambda_embed * (self.lm_recon_loss)

    def log_info(self):
        self.log_vars.append(("encode_loss", self.encode_loss))
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        self.log_vars.append(("LL_loss", self.LL_loss))
        self.log_vars.append(("recon_loss", self.recon_loss))
        self.log_vars.append(("Embed loss", self.Embed_loss))
        self.log_vars.append(("Embed loss same", self.positive_loss))
        self.log_vars.append(("Embed loss diff", self.negative_loss))
        self.log_vars.append(("Embed loss recon", self.lm_recon_loss))
        self.log_vars.append(("real_emotion_cls_loss", self.real_emotion_cls_loss))
        self.log_vars.append(("fake_emotion_cls_loss", self.fake_emotion_cls_loss))

    def crop_local_imgs(self, emo_imgs, fake_emo_imgs):
        points_batch = self.get_points()
        local_x_batch = []
        local_completion_batch = []
        for i in range(self.batch_size):
            x1, y1, x2, y2 = points_batch[i]
            local_x_batch.append(emo_imgs[i][y1:y2, x1:x2, :])
            local_completion_batch.append(fake_emo_imgs[i][y1:y2, x1:x2, :])
        # 真实样本
        local_x_batch = np.array(local_x_batch)
        # 生成的样本
        local_completion_batch = np.array(local_completion_batch)
        return local_x_batch, local_completion_batch

