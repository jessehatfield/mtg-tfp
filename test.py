import generate
import models
import plots

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import math
from matplotlib import pyplot as plt
import seaborn as sns


def test_ll_p_deck(session):
    p_deck_flat = np.array([[0.4, 0.3, 0.3],                                     # 0-0
                            [0.4, 0.3, 0.3], [0.4, 0.3, 0.3],                    # 1-0, 0-1
                            [0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3]])  # 2-0, 1-1, 0-2
    pairings_even = np.array([[40, 30, 30],                               # 0-0
                              [40, 30, 30], [40, 30, 30],                 # 1-0, 0-1
                              [40, 30, 30], [40, 30, 30], [40, 30, 30]])  # 2-0, 1-1, 0-2
    prob_delta = np.array([[0.0, 0.0, 0.0],                             # 0-0
                           [.02, 0, -.02], [-.02, 0, .02],              # 1-0, 0-1
                           [.02, 0, -.02], [0, 0, 0], [-.02, 0, .02]])  # 2-0, 1-1, 0-2
    n_pairings = tf.placeholder(dtype=tf.float32, shape=(6, 3))
    p = tf.placeholder(dtype=tf.float32, shape=(6, 3))
    ll = models.pairings_log_prob(n_pairings, tf.transpose(p))

    true_ll = session.run(ll, {n_pairings: pairings_even, p: p_deck_flat})
    print(true_ll)
    for i in range(1, 10):
        plus = p_deck_flat + (prob_delta * i)
        minus = p_deck_flat + (prob_delta * i)
        r1 = session.run(ll, {n_pairings: pairings_even, p: plus})
        r2 = session.run(ll, {n_pairings: pairings_even, p: minus})
        print(i, r1, r2)
    diff = []
    ll_modified = []
    modified = []
    s0 = np.array([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3],
                   [0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3]])
    ll0 = session.run(ll, {n_pairings: pairings_even, p: s0})
    d0 = math.sqrt(np.square(s0 - p_deck_flat).sum())
    ll_modified.append(ll0)
    modified.append(s0)
    diff.append(d0)
    for i in range(1000):
        sample = np.random.dirichlet([39.8, 30.1, 30.1], size=6)
        random_ll = session.run(ll, {n_pairings: pairings_even, p: sample})
        if random_ll > true_ll:
            print(random_ll)
            print(sample)
        d = math.sqrt(np.square(sample - p_deck_flat).sum())
        ll_modified.append(random_ll)
        modified.append(sample)
        diff.append(d)
    for i in range(100):
        print(diff[i], ll_modified[i], modified[i])
    sns.scatterplot(np.array(diff), np.array(ll_modified))
    plt.show()


def test_p_deck_approx(session):
    field = tf.placeholder(dtype=tf.float32, shape=(3,))
    matchups = tf.placeholder(dtype=tf.float32, shape=(3, 3))
    scores = tf.placeholder(dtype=tf.float32, shape=(5, 5))
    p_deck, p_opp_deck = models.approximate_p_deck(2, 3, field, matchups, scores)
    m = np.array([[.5, .7, .2], [.3, .5, .6], [.8, .4, .5]])
    f = np.array([.4, .3, .3])
    s = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    p = session.run(p_deck, {matchups: m, field: f, scores: s})
    print(p)


def test_ll(session):
    field = tf.placeholder(dtype=tf.float32, shape=(3,))
    matchups = tf.placeholder(dtype=tf.float32, shape=(3, 3))
    scores = tf.placeholder(dtype=tf.float32, shape=(5, 5))
    pairing_counts = tf.placeholder(dtype=tf.float32, shape=(3, 6))
    record_counts = tf.placeholder(dtype=tf.float32, shape=(3, 6))
    wait = tf.constant(100.0)
#    p_deck_given_record_approx, p_opp_deck_given_record = models.approximate_p_deck(2, 3, field, matchups, scores)
#    ll = models.pairings_log_prob(tf.transpose(pairing_counts), p_deck_given_record_approx)
    league = models.LeagueModel(3, 2)
    ll = league.log_prob_fn(tf.transpose(pairing_counts),
                            tf.transpose(record_counts))(field, generate.get_matchup_parameters(matchups), wait)

    m = np.array([[.5, .7, .2], [.3, .5, .6], [.8, .4, .5]])
    f = np.array([.4, .3, .3])
    s = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    n = np.array([[40, 38, 43, 36, 42, 41],
                  [30, 27, 33, 28, 29, 34],
                  [30, 36, 25, 35, 30, 24]])
    p = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    ll_true = session.run(ll, {matchups: m, field: f, scores: s, pairing_counts: p, record_counts: n})
    f_normalized = f / math.sqrt(np.square(f).sum())
    mf = session.run(generate.get_matchup_parameters(matchups), {matchups: m})
    print(mf)
    print(ll_true)
    print("Hold matchups constant:")
    f_modified = []
    ll_modified_field = [ll_true]
    sim_field = [f_normalized.dot(f_normalized).sum()]
    for i in range(1000):
        f_modified.append(np.random.dirichlet([1.0, 1.0, 1.0]))
        ll_modified_field.append(session.run(ll, {matchups: m, field: f_modified[-1], scores: s, pairing_counts: p, record_counts: n}))
        if ll_modified_field[-1] > ll_true:
            print(ll_modified_field[-1], f_modified[-1])
        normalized = f_modified[-1] / math.sqrt(np.square(f_modified[-1]).sum())
        cosine = normalized.dot(f_normalized).sum()
        sim_field.append(cosine)
    sim_field = np.array(sim_field)
    ll_modified_field = np.array(ll_modified_field)
    ax = sns.scatterplot(sim_field, ll_modified_field - ll_true)
    ax.hlines(0.0, xmin=-1.0, xmax=1.0, linestyle='dashed')
    ax.vlines(1.0, ymin=-1.0, ymax=1.0, linestyle='dashed')
    plt.show()
    print("Hold field constant:")
    m_free = tf.placeholder(dtype=tf.float32, shape=(3,))
    mm = generate.build_matchup_matrix(m_free, 3)
    m_modified = []
    dist_matchup = []
    ll_modified_matchup = []
    results = []
    # should be optimal:
    f0 = np.array([0.4, 0.8, 0.3])
    m_modified.append(session.run(mm, {m_free: f0}))
    ll0 = session.run(ll, {matchups: m_modified[-1], field: f, scores: s, pairing_counts: p, record_counts: n})
    d0 = math.sqrt(np.square(f0 - mf).sum())
    dist_matchup.append(d0)
    ll_modified_matchup.append(ll0)
    results.append((ll0, m_modified[-1], d0))
    for i in range(1000):
        free = np.random.beta(12, 12, 3)
        m_modified.append(session.run(mm, {m_free: free}))
        ll_modified = session.run(ll, {matchups: m_modified[-1], field: f, scores: s, pairing_counts: p, record_counts: n})
        if ll_modified > ll_true:
            print(ll_modified, m_modified[-1])
        dist = math.sqrt(np.square(free - mf).sum())
        dist_matchup.append(dist)
        ll_modified_matchup.append(ll_modified)
        results.append((ll_modified, m_modified[-1], dist))
    results.sort(key=lambda x: x[0])
    for i in range(10):
        print(results[i])
    print('...')
    for i in range(10):
        print(results[i-10])
    dist_matchup = np.array(dist_matchup)
    ll_modified_matchup = np.array(ll_modified_matchup)
    ax = sns.scatterplot(dist_matchup, ll_modified_matchup - ll_true)
    ax.hlines(0.0, xmin=-1.0, xmax=1.0, linestyle='dashed')
    ax.vlines(0.0, ymin=-1.0, ymax=1.0, linestyle='dashed')
    plt.show()


if __name__ == "__main__":
    session = tf.Session()
    test_ll_p_deck(session)
    test_p_deck_approx(session)
    test_ll(session)
