import tensorflow as tf
from jrstnets import LSTMNetFactory

#real data: X
#lstmnet1
#lstmnet2, bidirectional?

# G_sample = lstmnet1.output(X)
# D_real = lstmnet2.output(X)
# D_fake = lstmnet2.output(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# variables in generator: G_vars
# variables in discriminator: D_vars

# D_optimizer = tf.train.someoptimizer(D_lr).minimize(D_loss, var_list = G_vars)
# G_optimizer = tf.train.someoptimizer(G_lr).minimize(G_loss, var_list = D_vars)



