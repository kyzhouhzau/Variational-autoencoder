#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Tensorflow Probability Authors.
Licensed under the Apche License, Version 2.0 (the "License").
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

该脚本用于编码依存关系。
数据输入格式是【单词，依存关系，单词，依存关系。。。。。】
"""

import functools
import os

from absl import flags,logging
import numpy as np
import tensorflow as tf
import tensorflow_probability  as tfp
from six.moves import urllib
tfd = tfp.distributions
tf.enable_eager_execution()
TEXT_SHAPE = [100,9,1]
flags.DEFINE_integer("base_depth",default=32,help="base depth for layer")
flags.DEFINE_float("learning_rate",default=0.001,help="Initial learning rate.")
flags.DEFINE_integer("max_steps",default=500,help="Number of training steps to run.")
flags.DEFINE_integer("latent_size",default=16,help="Number of dimension in he latent code (z).")
flags.DEFINE_string("activation",default="leaky_relu",help="Activation function for all hidden layers.")
flags.DEFINE_integer("batch_size",default=32,help="Batch size.")
flags.DEFINE_integer("n_samples",default=16,help = "Number of samples to use in encoding.")
flags.DEFINE_integer("mixture_components",default=100,help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")
flags.DEFINE_bool("analytic_kl",default=False,help="Whether or not to use the analytic version of the KL. When set to "
         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
         "then you must also specify `mixture_components=1`.")
flags.DEFINE_string("data_dir",default="AutorEncoder/TEST_TMPDIR/vae/data",
                    help="Directory where data is stored (if using real data).")

flags.DEFINE_string("model_dir",default="E:\AutorEncoder/Model/vae/",
                    help="Directory to put the model's fit.")

flags.DEFINE_integer("viz_steps",default=500,help="Frequency at which to save visualizations.")

flags.DEFINE_bool("fake_data",default=False,help="If true, uses fake data instead of MNIST.")

flags.DEFINE_bool("delete_existing",default=True,help = "If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS

def _softplus_inverse(x):
    """log(exp(x)-1)"""
    return tf.math.log(tf.math.expm1(x))

def make_encoder(activation,latent_size,base_depth):
    "Creates the encoder function"
    #functools.partial()高阶函数，用于部分应用一个函数。
    conv = functools.partial(
        tf.keras.layers.Conv2D,padding="SAME", activation=activation
    )
    #base_depth=32 latent_size=16
    #这里在编码过程中不采用same是因为文本序列的特殊性
    encoder_net = tf.keras.Sequential([
        conv(2*latent_size,(1,1),(1,1)),#32 100 9 32
        conv(4*latent_size,(100,2),(1,1),padding="VALID"),#32 1 8 64
        conv(4*latent_size,(1,2),(1,2),padding="VALID"),#32 1 4 64
        conv(8*latent_size,(1,3),(1,2),padding="VALID"),#32 1 1 128

        #5层卷积，前4层卷积后的图像尺寸不变。最后一层卷积改变了尺寸。
        #再展开后，用全连接层进行变换batch_size*(2*latent_size)
        tf.keras.layers.Flatten(),#[
        tf.keras.layers.Dense(2*latent_size,activation=None)#
    ])
    def encoder(images):
        #print(images) shape=(32, 100, 30, 1)
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
            loc = net[...,:latent_size],
            scale_diag = tf.nn.softplus(net[...,latent_size:]+_softplus_inverse(1.0)),#这一句不理解，
            name="code"
        )
    return encoder

def make_decoder(activation,laten_size,output_shape,base_depth):
    "create decode function"
    # output_shape..(100,30,1)
    # input shape 32 100 1 16
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose,padding="SAME",activation=activation
    )
    conv = functools.partial(
        tf.keras.layers.Conv2D,padding="SAME",activation=activation
    )
    #第二个参数是卷积核大小，第三个参数是stride
    decoder_net = tf.keras.Sequential([
        deconv(2*base_depth,(1,1)),#32 1 1 64
        deconv(2*base_depth,(100,2),(1,1),padding="VALID"),
        deconv(2*base_depth,(1,2),(1,2),padding="VALID"),
        deconv(2*base_depth,(1,3),(1,2),padding="VALID"),

        conv(output_shape[-1],(100,1),activation=None,)# 32 100 9 1
    ])
    def decoder(codes):
        original_shape = tf.shape(input=codes)#
        codes = tf.reshape(codes,(-1,1,1,laten_size))
        #print(codes)  shape=(32, 1, 1, 16)
        logits = decoder_net(codes)#
        print(logits)
        logits = tf.reshape(logits,shape = tf.concat([original_shape[:-1],output_shape],axis=0))
        return tfd.Independent(tfd.Binomial(logits=logits,total_count=100),
                                reinterpreted_batch_ndims=len(output_shape),
                               name="output")
    return decoder




def make_mixture_prior(latent_size,mixture_components):
    "Crete the mixture of Gaussians prior distribution."
    if mixture_components==1:
        return tfd.MultivariateNormalDiag(
            loc = tf.zeros([latent_size]),
            scale_identity_multiplier=1.0
        )
    loc = tf.get_variable(
        name="loc",shape = [mixture_components,latent_size]
    )
    raw_scale_diag = tf.get_variable(
        name="raw_scale_diag",shape = [mixture_components,latent_size]
    )
    mixture_logits = tf.get_variable(
        name = "mixture_logits",shape=[mixture_components]
    )
    return tfd.MixtureSameFamily(
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=tf.nn.softplus(raw_scale_diag)
        ),
        mixture_distribution = tfd.Categorical(logits=mixture_logits),#表示由mixture_logits个高斯分布构成
        name = "prior"
    )

def model_fn(features,labels,mode,params,config):
    if params["analytic_kl"] and params["miture_components"]!=1:
        raise NotImplementedError(
            "Using 'analytic_k1' is only supported when `mixture_components=1`"
            "since there's no closed from otherwise!"
        )
    encoder = make_encoder(params["activation"],
                           params["latent_size"],
                           params["base_depth"])

    decoder = make_decoder(params["activation"],
                           params["latent_size"],
                           TEXT_SHAPE,
                           params["base_depth"])

    #就是要近似的那个后验，用于求ELBO
    latent_prior = make_mixture_prior(params["latent_size"],
                                      params["mixture_components"])
    #这个是中间层包含均值和方差。从该层采样后构成近似后验
    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    #这里是解码层
    decoder_likelihood = decoder(approx_posterior_sample)
    ################################################################################
    ####################################构建ELBO#####################################
    ################################################################################
    print("decode_likehood",decoder_likelihood)
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(input_tensor=distortion)
    tf.summary.scalar("distortion",avg_distortion)

    print("################")
    if params["analytic_kl"]:
        rate = tfd.kl_divergence(approx_posterior,latent_prior)
    else:
        #E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)]
        rate = (approx_posterior.log_prob(approx_posterior_sample)-
                latent_prior.log_prob(approx_posterior_sample))
    avg_rate = tf.reduce_mean(input_tensor=rate)
    print("################")

    tf.summary.scalar("rate",avg_rate)
    print(rate)
    print(distortion)
    elbo_local = -(rate+distortion)
    print("################")

    elbo = tf.reduce_mean(input_tensor=elbo_local)
    loss = -elbo
    tf.summary.scalar("elbo",elbo)
    ################################################################################
    ####################################构建elbo#####################################
    ################################################################################

    #decode sample from the prior for visulization.
    ###################可以用于考察分布在如何变化########################
    #对p(z)采样并解码可以生成一张新的图片该图片拥有其他图片的特征但是不完全相同（在图像中）
    #用于新的图像生成
    random_image = decoder(latent_prior.sample(16))
    ###############################################################
    # minimizing the -ELBO.
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(
        params["learning_rate"],global_step,params["max_steps"]
    )
    tf.summary.scalar("learning_rate",learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss,global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo":tf.metrics.mean(elbo),
            "rate":tf.metrics.mean(avg_rate),
            "distortion":tf.metrics.mean(avg_distortion)
        }
    )



def build_fake_input_fns(batch_size):
    "builds fake training testing data!"
    random_sample = np.random.rand(batch_size,*TEXT_SHAPE).astype("float32")
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(random_sample).map(lambda row:(row,0)).batch(batch_size).repeat()
        return tf.data.make_one_shot_iterator(dataset).get_next()

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(random_sample).map(lambda row: (row,0)).batch(batch_size)
        return tf.data.make_one_shot_iterator(dataset).get_next()

    return train_input_fn,eval_input_fn

#在这里传入argv是必须的由于适用absl
def main(argv):
    del argv
    params = FLAGS.flag_values_dict()
    #getattr()获得tf.nn中的激活函数属性
    params["activation"] = getattr(tf.nn,params["activation"])
    if FLAGS.delete_existing and tf.io.gfile.exists(FLAGS.model_dir):
        logging.warning("Deleteing old log directory at {}".format(
            FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)


    train_input_fn,eval_input_fn = build_fake_input_fns(FLAGS.batch_size)


    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config = tf.estimator.RunConfig(
            model_dir = FLAGS.model_dir,
            save_checkpoints_steps = FLAGS.viz_steps
        )
    )

    for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
        train_r = estimator.train(train_input_fn,steps=FLAGS.viz_steps)
        print(train_r.elbo)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)

if __name__=="__main__":
    tf.app.run()


