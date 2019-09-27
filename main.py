import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from IPython.core.pylabtools import figsize

session = tf.Session()

real_dist = tfp.distributions.Bernoulli(probs=0.25, dtype=tf.int32)

# Sample from the real distribution
n = tf.constant([0, 1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 2000])
trials = real_dist.sample(n[-1])
# Get counts for varying numbers of samples
padded = tf.pad(trials, tf.constant([[1, 0, ]]), "CONSTANT")
counts = tf.gather(tf.cumsum(padded), n)

# Construct a posterior distribution for the original probability
#alpha = tf.cast(1 + counts, tf.float32)
#beta = tf.cast(1 + n - counts, tf.float32)
#posterior = tfp.distributions.Beta(concentration1=alpha, concentration0=beta)
#p = tf.linspace(start=0.0, stop=1.0, num=100)
#prob = tf.transpose(posterior.prob(p[:, tf.newaxis]))
# Compute probabilities
#[n_, p_, prob_, counts_] = session.run([n, p, prob, counts])

# Infer a posterior distribution for the original probability:
# define model
prior = tfp.distributions.Beta(1, 1)
init = [tf.constant([0.2] * 12, tf.float32, name="init_p")]
bijectors = [tfp.bijectors.Sigmoid()]
def ll(p):
    successes = tfp.distributions.Binomial(tf.cast(n, dtype=tf.float32), probs=p)
    return successes.log_prob(tf.cast(counts, dtype=tf.float32))

# configure
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(name='step_size',
                                initializer=tf.constant(0.05, tf.float32),
                                trainable=False,
                                use_resource=True)
# sample
[posterior], results = tfp.mcmc.sample_chain(num_results=10000, num_burnin_steps=10000,
                                              current_state=init,
                                              kernel=tfp.mcmc.TransformedTransitionKernel(
                                                  inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                                                      target_log_prob_fn=ll,
                                                      num_leapfrog_steps=2,
                                                      step_size=step_size,
                                                      step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
                                                      state_gradients_are_stopped=True),
                                                  bijector=bijectors))
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
session.run([init_g])
session.run([init_l])
[samples_, n_, counts_, results_] = session.run([posterior, n, counts, results])
print("Accepted:", results_.inner_results.is_accepted.mean())
print("Final step size:", results_.inner_results.extra.step_size_assign[-100:].mean())


# Plot posteriors
plt.figure(figsize(16, 9))
p = tf.linspace(start=0.0, stop=1.0, num=100)
for i in range(len(n_)):
    sx = plt.subplot(len(n_)/2, 2, i+1)
    plt.xlabel("$P(success)$") if i in [0, len(n_)-1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
#    plt.plot(p_, prob_[i], label="observed {} successes in {} trials".format(counts_[i], n_[i]))
#    plt.fill_between(p_, 0, prob_[i], color="#5DA5DA", alpha=0.4)
#    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
#    plt.hist(samples_[:,i], histtype='stepfilled', bins=30, alpha=0.85,
#             label="observed {} successes in {} trials".format(counts_[i], n_[i]),
#             range=[0,1])
    sns.distplot(samples_[:,i], hist = False, kde = True, rug = True)
#    leg = plt.legend()
#    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)

plt.suptitle("Posterior probability updates over time", y=1.02, fontsize=14)
plt.tight_layout()
