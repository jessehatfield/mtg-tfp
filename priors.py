import tensorflow as tf
import tensorflow_probability as tfp


def matchup(alpha=10):
    return tfp.distributions.Beta(alpha, alpha)


def field(k, alpha=1):
    concentration = tf.ones(k) * alpha
    return tfp.distributions.Dirichlet(concentration)
