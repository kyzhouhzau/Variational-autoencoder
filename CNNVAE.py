#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Tensorflow Probability Authors.
Licensed under the Apche License, Version 2.0 (the "License").
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
"""

import functools
import os

from absl import flags,logging
import numpy as np
import tensorflow as tf
import tensorflow_probability  as tfp
from six.moves import urllib
tfd = tfp.distributions

IMAGE_SHAPE = [28,28,1]
flags.DEFINE_integer("base_depth",default=32,help="base depth for layer")
flags.DEFINE_float("learning_rate",default=0.001,help="Initial learning rate.")
flags.DEFINE_integer("max_steps",default=5000,help="Number of training steps to run.")
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
flags.DEFINE_string("data_dir",default="E:\AutorEncoder/TEST_TMPDIR/vae/data",
                    help="Directory where data is stored (if using real data).")

flags.DEFINE_string("model_dir",default="E:\AutorEncoder/Model/vae/",
                    help="Directory to put the model's fit.")

flags.DEFINE_integer("viz_steps",default=500,help="Frequency at which to save visualizations.")

flags.DEFINE_bool("fake_data",default=False,help="If true, uses fake data instead of MNIST.")

flags.DEFINE_bool("delete_existing",default=False,help = "If true, deletes existing `model_dir` directory.")

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
    encoder_net = tf.keras.Sequential([
        conv(base_depth,5,1),#16 28 28 32
        conv(base_depth,5,2),#16 14 14 32
        conv(2*base_depth,5,1),#16 14 14 64
        conv(2*base_depth,5,2),#16 7 7 64
        conv(4*latent_size,7,padding="VALID"),#16 1 1 64
        #5层卷积，前4层卷积后的图像尺寸不变。最后一层卷积改变了尺寸。
        #再展开后，用全连接层进行变换batch_size*(2*latent_size)
        tf.keras.layers.Flatten(),#[16*64] [64*2*16]
        tf.keras.layers.Dense(2*latent_size,activation=None)#16 2*16
    ])

    def encoder(images):
        images = 2 * tf.cast(images, dtype=tf.float32) - 1
        net = encoder_net(images)#16 2*16
        return tfd.MultivariateNormalDiag(
            loc = net[...,:latent_size],
            scale_diag = tf.nn.softplus(net[...,latent_size:]+_softplus_inverse(1.0)),#这一句不理解，
            name="code"
        )
    return encoder

def make_decoder(activation,laten_size,output_shape,base_depth):
    "create decode function"
    # output_shape..(28,28,1)
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose,padding="SAME",activation=activation
    )
    conv = functools.partial(
        tf.keras.layers.Conv2D,padding="SAME",activation=activation
    )
    #第二个参数是卷积核大小，第三个参数是stride
    decoder_net = tf.keras.Sequential([
        deconv(2*base_depth,7,padding="VALID"),#16 7 7 64
        deconv(2*base_depth,5),#16 7 7 64
        deconv(2*base_depth,5,2),#16 14 14 64
        deconv(base_depth,5),#16 14 14 32
        deconv(base_depth,5,2),#16 28 28 32
        deconv(base_depth,5),#16 28 28 32
        #这里的depth变成1
        conv(output_shape[-1],5,activation=None)#16 28 28 1
    ])
    def decoder(codes):
        original_shape = tf.shape(input=codes)#16 32
        codes = tf.reshape(codes,(-1,1,1,laten_size))#16 1 1 32
        logits = decoder_net(codes)#16 28 28 1
        logits = tf.reshape(
            logits,shape = tf.concat([original_shape[:-1],output_shape],axis=0)#16 * 28*28*1
        )
        return tfd.Independent(tfd.Bernoulli(logits),
                               reinterpreted_batch_ndims=len(output_shape),
                               name="image")
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
        mixture_distribution = tfd.Categorical(logits=mixture_logits),
        name = "prior"
    )

def pack_images(images,rows,cols):
    shape = tf.shape(input=images)#16 28 28 1
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images,(-1,width,height,depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows,batch)
    cols = tf.minimum(batch//rows,cols)
    images = images[:rows*cols]
    images  = tf.reshape(images,(rows,cols,width,height,depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    images = tf.reshape(images,[1,rows*width,cols*height,depth])
    return images
#max_outpus：要生成图像的最大批处理元素数。
def image_title_summary(name,tensor,rows=8,cols=8):
    # max_outputs要生成图像的最大批处理元素数
    tf.summary.image(name,pack_images(tensor,rows,cols),max_outputs=1)

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
                           IMAGE_SHAPE,
                           params["base_depth"])

    latent_prior = make_mixture_prior(params["latent_size"],
                                      params["mixture_components"])
    image_title_summary("input",tf.cast(features,dtype=tf.float32),rows=1,cols=16)
    print(features)
    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    decoder_likelihood = decoder(approx_posterior_sample)
    image_title_summary(
        "recon/sample",
        tf.cast(decoder_likelihood.sample()[:3,:16],tf.float32),
        rows = 3,
        cols = 16
    )
    image_title_summary(
        "recon/mean",
        decoder_likelihood.mean()[:3,:16],
        rows=3,
        cols=16
    )
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(input_tensor=distortion)
    tf.summary.scalar("distortion",avg_distortion)

    if params["analytic_kl"]:
        rate = tfd.kl_divergence(approx_posterior,latent_prior)
    else:
        #E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)]
        rate = (approx_posterior.log_prob(approx_posterior_sample)-
                latent_prior.log_prob(approx_posterior_sample))
    avg_rate = tf.reduce_mean(input_tensor=rate)

    tf.summary.scalar("rate",avg_rate)

    elbo_local = -(rate+distortion)

    elbo = tf.reduce_mean(input_tensor=elbo_local)
    loss = -elbo
    tf.summary.scalar("elbo",elbo)

    importance_weighted_elbo = tf.reduce_mean(
        input_tensor = tf.reduce_logsumexp(input_tensor=elbo_local,axis=0)-
        tf.math.log(tf.cast(params["n_samples"],dtype=tf.float32))
    )
    tf.summary.scalar("elbo/importance_weighted",importance_weighted_elbo)

    #decode sample from the prior for visulization.
    random_image = decoder(latent_prior.sample(16))
    image_title_summary(
        "random/sample",
        tf.cast(random_image.sample(),dtype=tf.float32),
        rows=4,
        cols=4
    )
    image_title_summary("random/mean",random_image.mean(),rows=4,cols=4)

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
            "elbo/importance_weighted":tf.metrics.mean(importance_weighted_elbo),
            "rate":tf.metrics.mean(avg_rate),
            "distortion":tf.metrics.mean(avg_distortion)
        }
    )

ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"

def download(directory,filename):
    filepath = os.path.join(directory,filename)
    if tf.io.gfile.exists(directory):
        return filepath
    else:
        tf.io.gfile.makedirs(directory)
    url = os.path.join(ROOT_PATH,filename)
    urllib.request.urlretrieve(url,filepath)
    return filepath

def static_minist_dataset(directory,split_name):
    "Return binary static MNIST tf.data.Dataset."
    amat_file = download(directory,FILE_TEMPLATE.format(split = split_name))
    #dataset对应文件中得一行
    dataset = tf.data.TextLineDataset(amat_file)
    str_to_arr = lambda string:np.array([c==b"1" for c in string.split()])
    def _paser(s):
        booltensor = tf.py_func(str_to_arr,[s],tf.bool)
        reshaped = tf.reshape(booltensor,[28,28,1])
        return tf.cast(reshaped,dtype=tf.float32),tf.constant(0,tf.int32)
    return dataset.map(_paser)

def build_fake_input_fns(batch_size):
    "builds fake MNIST-style data for unit testing"
    random_sample = np.random.rand(batch_size,*IMAGE_SHAPE).astype("float32")
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(random_sample).map(lambda row:(row,0)).batch(batch_size).repeat()
        return tf.data.make_one_shot_iterator(dataset).get_next()

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(random_sample).map(lambda row: (row,0)).batch(batch_size)
        return tf.data.make_one_shot_iterator(dataset).get_next()

    return train_input_fn,eval_input_fn

def build_input_fns(data_dir,batch_size):
    ""
    def train_input_fn():
        dataset = static_minist_dataset(data_dir,"train")
        dataset = dataset.shuffle(50000).repeat().batch(batch_size)

        return tf.data.make_one_shot_iterator(dataset).get_next()

    def eval_input_fn():
        eval_dataset = static_minist_dataset(data_dir,"vaild")
        eval_dataset = eval_dataset.batch(batch_size)
        return tf.data.make_one_shot_iterator(eval_dataset).get_next()

    return train_input_fn,eval_input_fn

def main(argv):
    del argv
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn,params["activation"])
    if FLAGS.delete_existing and tf.io.gfile.exists(FLAGS.model_dir):
        logging.warning("Deleteing old log directory at {}".format(
            FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)

    if FLAGS.fake_data:
        train_input_fn,eval_input_fn = build_fake_input_fns(FLAGS.batch_size)
    else:
        train_input_fn,eval_input_fn = build_input_fns(FLAGS.data_dir,FLAGS.batch_size)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config = tf.estimator.RunConfig(
            model_dir = FLAGS.model_dir,
            save_checkpoints_steps = FLAGS.viz_steps
        )
    )

    for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
        estimator.train(train_input_fn,steps=FLAGS.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__=="__main__":
    tf.app.run()


