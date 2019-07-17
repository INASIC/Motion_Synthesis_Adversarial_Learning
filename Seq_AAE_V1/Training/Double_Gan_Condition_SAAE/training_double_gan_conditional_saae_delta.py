import sys
import os

# print os.getcwd()


root_path = os.getcwd()[:-45]  # Gets the path above the root directory
print root_path
sys.path.append(root_path)
from Seq_AAE_V1.datasets.dataset import Emilya_Dataset

# Edited:
from Seq_AAE_V1.double_gan_conditional_saae import Double_GAN_Conditional_SAAE

# from Seq_AAE_V1.models.Double_GAN_Conditional_SAAE.double_gan_conditional_saae import Double_GAN_Conditional_SAAE

import tensorflow as tf

tf.keras.backend.clear_session()  # For easy reset of notebook state.

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

# with sess.as_default():
#   assert tf.compat.v1.get_default_session() is sess

# with sess.as_default()

# tf.Session()


# =============== ORIGINAL.... FULL TRAINING ========================
# dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,sampling_interval=None,with_velocity=True,
#                              number=None,nb_valid=None,nb_test=None)
#
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# # with tf.Session() as sess:
# #     init = tf.global_variables_initializer()
# #     sess.run(init)
#
# model = Double_GAN_Conditional_SAAE(latent_dim=50,
#                                     latent_activation='tanh',
#                                     latent_BN=False,
#                                     hidden_dim_enc_list=[100,100],
#                                     activation_enc_list=['tanh','tanh'],
#                                     hidden_dim_dec_list=None,
#                                     activation_dec_list=None,
#                                     hidden_dim_dis_list=[100,40],
#                                     hidden_dim_classifier_list=[100,40],
#                                     activation_dis_list=['relu','relu'],
#                                     activation_classifier_list=['relu','relu'],
#                                     dropout_dis_list=[0.0,0.0],
#                                     dropout_classifier_list=[0.0,0.0],
#                                     batch_size=200,
#                                     max_epoch=401,
#                                     optimiser_autoencoder='rmsprop',
#                                     optimiser_dis='adam',
#                                     optimiser_classifier='adam',
#                                     lr_autoencoder=0.001, lr_dis=0.001,
#                                     decay_autoencoder=0.0, decay_dis=0.0,
#                                     lr_classifier=0.001,decay_classifier=0.0,
#                                     momentum_autoencoder=0.0, momentum_dis=0.0,
#                                     momentum_classifier=0.0,
#                                     prior_noise_type='Gaussian',
#                                     loss_weights=[1.0,0.001,0.0005],
#                                     train_disc=True,
#                                     train_classifier=True,
#                                     checkpoint_epochs=2,
#                                     symetric_autoencoder=False,
#                                     nb_to_generate=30,
#                                     custom_loss=True,
#                                     loss_weight_mse_v=100.0,
#                                     condition_activity_or_emotion=2,
#                                     nb_label=8,
#                                     fully_condition=True,
#                                     embedding_dim=0,
#                                     is_annealing_beta=True,
#                                     beta_anneal_rate=0.1,
#                                     bias_beta=9.0)

# =============== /model/.... QUICK TRAINING ========================
dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,
                             sampling_interval=None,
                             with_velocity=True,
                             number=200,nb_valid=200,nb_test=200)

model = Double_GAN_Conditional_SAAE(latent_dim=50,latent_activation='tanh',latent_BN=False,
                                hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                                hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'],
                                hidden_dim_dis_list=[100,40],activation_dis_list=['relu','relu'],
                                dropout_dis_list=[0.0,0.0],batch_size=20,max_epoch=100,
                                optimiser_autoencoder='rmsprop',optimiser_dis='sgd',
                                lr_autoencoder=0.001,lr_dis=0.01,decay_autoencoder=0.0,decay_dis=0.0,
                                momentum_autoencoder=0.0,momentum_dis=0.0,
                                prior_noise_type='Gaussian',
                                loss_weights=[1.0,0.001,0.001],
                                train_disc=True,
                                train_classifier=True,
                                checkpoint_epochs=10,
                                custom_loss=True,symetric_autoencoder=False,
                                nb_to_generate=10,
                                condition_activity_or_emotion=1,
                                nb_label=8,fully_condition=False,
                                is_annealing_beta=True,
                                beta_anneal_rate=0.1,
                                bias_beta=9.)

model.training(dataset_obj)
