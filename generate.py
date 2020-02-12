"""Functions for generating data."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.contrib.distributions import fill_triangular, fill_triangular_inverse

import priors
import simulation


def build_matchup_matrix(p_win_free, k: int, name="M"):
    """Given k*(k-1)/2 free parameters, construct the full k*k matchup matrix M
    k = ceil(sqrt(2n))

    Args:
        p_win_free: Tensor representing independent win probabilities. Should
            have one dimension with (k*(k-1)/2) entries, where k is the number
            of objects to be paired up. Should have real-valued entries (0,1).
        k: integer representing the total number of objects.
        name: name of the returned tensor

    Returns:
         A tensor of shape (k,k)
    """
    p_mirror_mat = tf.eye(k, k) * 0.5
    mat = fill_triangular(p_win_free, upper=False)
    # pad the last two dimensions, leaving any others alone
    padding = [[0,0] for i in range(mat.get_shape().ndims)]
    padding[-2] = [1, 0]
    padding[-1] = [0, 1]
    p_win_mat = tf.pad(mat, padding) + p_mirror_mat
    # transpose the last two dimensions, leaving any others alone
    permute = [i for i in range(p_win_mat.get_shape().ndims)]
    permute[-2] = permute[-2] + 1
    permute[-1] = permute[-1] - 1
    p_lose_mat = 1 - tf.transpose(p_win_mat, perm=permute)
    return tf.where(p_win_mat + p_lose_mat >= 1, p_win_mat, p_lose_mat, name=name)


def get_matchup_parameters(matchup_matrix):
    """Given a k*k matchup matrix M, extract the k*(k-1)/2 free parameters as a vector."""
    # Truncate the last two dimensions
    shape = matchup_matrix.get_shape()
    begin = [0 for i in range(shape.ndims)]
    size = [shape[i] for i in range(shape.ndims)]
    k = size[-1].value
    size[-2] -= 1
    size[-1] -= 1
    begin[-2] = 1 # drop the first row (and the last column)
    # Undocumented feature: fill_triangular_inverse will add non-triangular entries to other entries,
    # so we need to set them to zero first
    banded = tf.matrix_band_part(matchup_matrix, -1, 0)  # still need to get rid of diagonal
    zeroed = banded - (tf.eye(k, k) * 0.5)  # now only entries below the diagonal will be non-zero
    sliced = tf.slice(zeroed, begin, size)
    return fill_triangular_inverse(sliced, upper=False)


def free_to_matrix(k: int):
    """Get a list of 2D indices in the k*k matchup matrix corresponding to the k*(k-1)/2 free matchup parameters.

    Args:
        k: integer representing the total number of objects.

    Returns:
        a tensor of shape (k*(k-1)/2, 2) which evaluates to a list of [i,j] pairs.
    """
    col_ids = tf.constant(np.broadcast_to(np.arange(k), (k, k)), dtype=tf.int32)
    row_ids = tf.transpose(col_ids)
    zero_upper = fill_triangular(tf.ones(int(k*(k+1)/2), dtype=tf.int32), upper=False) - tf.eye(k, dtype=tf.int32)
    row_lower = tf.multiply(row_ids, zero_upper)
    col_lower = tf.multiply(col_ids, zero_upper)
    begin = [1, 0]
    size = [k-1, k-1]
    row_flat = fill_triangular_inverse(tf.slice(row_lower, begin, size), upper=False)
    col_flat = fill_triangular_inverse(tf.slice(col_lower, begin, size), upper=False)
    indices = tf.stack([row_flat, col_flat], axis=1)
    return indices


def score_matrix_k(delta, n_scores, p_score):
    """Compute P(opp.score | pl.score, max score difference)"""
    p_score_score_k = np.zeros((n_scores, n_scores), dtype=np.float32)
    for j in range(n_scores):
        # P(opp.score=s_i | pl.score=s_j, delta)
        #   = P(opp.score=s_i) / sum[d=-delta...delta]{ P(opp.score=s_j+d) }
        #   = P(s_i) / sum[d=-delta...delta]{ P(s_j+d) }
        valid = [i for i in range(j-delta, j+delta+1) if 0 <= i < n_scores]
        total = sum([p_score[i] for i in valid])
        for i in valid:
            p_score_score_k[i, j] = p_score[i] / total
    return p_score_score_k


def build_score_matrix(wait_time, p_find_given_delta, p_score, r: int):
    """Given a scalar parameter and maximum number of rounds, compute the full (2r+1)*(2r+1) score matrix S.
    
    S[i,j] represents the probability of getting paired against a player with score i given that you have score j,
    where a player's score is (wins - losses), under the model where the pairing algorithm initially attempts to find
    an opponent with the same score but iteratively relaxes the maximum score difference by 1 after passing up on
    approximately (wait_time) players outside the current threshold.
    
    Args:
        wait_time: 1-D Tensor representing possible wait times before relaxing the threshold in terms of average number
            of players seen in that interval.
        p_find_given_delta
        p_score
        r: integer representing the maximum number of rounds.

    Returns:
        A tensor of shape n*(2r+1)*(2r+1), the first dimension corresponds to the wait_time entries and the other
            dimensions' indices 0...2r correspond to score (win-loss) -r...+r.
    """
    n_scores = int((2 * r) + 1)
    wait_time = tf.reshape(wait_time, (-1, 1, 1))
    p_find_given_delta = tf.reshape(p_find_given_delta, (-1, n_scores, n_scores))
    expected_successes = wait_time * p_find_given_delta
    p_success_given_delta_score = 1 - tf.exp(-expected_successes)
    p_score_given_score = np.zeros((1, n_scores, n_scores))
    p_deltas = []
    for delta in range(0, n_scores):
        if len(p_deltas) == 0:
            # P(D=0) = P(success|D=0)
            p_delta_given_score = p_success_given_delta_score[:, :, delta]
        else:
            # P(D=1) = (1-P(D=0)) * P(success|D=1)
            # P(D=2) = (1-P(D=0)-P(D=1)) * P(success|D=2)
            p_delta_given_score = tf.multiply(1.0 - tf.add_n(p_deltas), p_success_given_delta_score[:, :, delta])
        p_deltas.append(p_delta_given_score)
        # P(S1 | S2) = sum[D]{P(S1 | S2, D) * P(D | S2)}
        p_score_given_score_delta = tf.reshape(score_matrix_k(delta, n_scores, p_score), (1, n_scores, n_scores))
        p_delta_given_score_reshaped = tf.reshape(p_delta_given_score, (-1, 1, n_scores))
        p_score_given_score += tf.multiply(p_score_given_score_delta, p_delta_given_score_reshaped)
    return p_score_given_score


def gen_matchups(n_archetypes: int, matchup_prior):
    """Sample matchup percentages given a number of archetypes and a prior."""
    n_matchups = int(n_archetypes * (n_archetypes-1) / 2)
    # Sample n*(n-1)/2 free parameters, then construct the full n*n matchup matrix M
    p_win = matchup_prior.sample(n_matchups)
    matchups = build_matchup_matrix(p_win, n_archetypes)
    return p_win, matchups


def gen_field(n_archetypes: int, field_prior):
    """Sample a field distribution given a number of archetypes and a prior."""
    return field_prior.sample(n_archetypes)


def rv_match_counts(field_vec: tf.Tensor, n: int):
    """Distribution of pairing counts.

    Define the distribution of independent pairings given a field
    distribution and total number of matches. Returns a flat tensor; unstack
    to get a square matrix of counts."""
    joint_prob_mat = tf.tensordot(field_vec, field_vec, 0)
    joint_prob_vec = tf.reshape(joint_prob_mat, [-1])
    return tfp.distributions.Multinomial(n, probs=joint_prob_vec)


def rv_deck_counts(field_vec: tf.Tensor, n: int):
    """Distribution of deck counts.

    Define the distribution of independent deck observations given a field distribution."""
    return tfp.distributions.Multinomial(n, probs=field_vec)


def rv_outcomes(match_counts, win_prob):
    """Define the distribution of match outcomes given match counts and matchup
    win probabilities."""
    p_outcome = wins_to_outcomes(win_prob)
    return tfp.distributions.Multinomial(match_counts, probs=p_outcome)


def rv_pairing_counts(record_totals, p_deck_given_record):
    return tfp.distributions.Multinomial(
        tf.cast(record_totals, dtype=tf.float32),
        probs=tf.linalg.transpose(p_deck_given_record))


def wins_to_outcomes(wins, total=1):
    """Convert a matrix of win statistics to a rank-3 tensor whose last
    dimension has shape two by stacking [wins,  (total-wins)]."""
    return tf.stack([wins, total-wins], axis=2)


def outcomes_to_wins(outcomes):
    """Take just the first matrix from an outcomes stack, representing wins
    only."""
    return tf.unstack(outcomes, axis=2)[0]


def gen_sample_data(
        session: tf.Session,
        n_archetypes,
        n_matches,
        matchup_prior=priors.matchup(),
        field=None):
    if field is None:
        field = priors.field(n_archetypes).sample()
    p_win, matchups = gen_matchups(n_archetypes, matchup_prior)
    pairings = tf.reshape(rv_match_counts(field, n_matches).sample(), matchups.shape)
    win_counts = outcomes_to_wins(rv_outcomes(pairings, matchups).sample())
    return session.run({
        'field': field,
        'matchups': matchups,
        'matchups_free': p_win,
        'match_counts': pairings,
        'match_wins': win_counts
    })


def gen_matches(field, matchups, n_matches):
    sample = {}
    match_counts = np.zeros((len(field), len(field)), dtype=np.float32)
    wins = np.zeros((len(field), len(field)), dtype=np.float32)
    for i in range(n_matches):
        decks = np.random.choice(len(field), size=2, replace=False, p=None)
        d1 = decks[0] if decks[0] <= decks[1] else decks[1]
        d2 = decks[1] if decks[0] <= decks[1] else decks[0]
        match_counts[d1][d2] += 1.0
        if np.random.random() < matchups[d1][d2]:
            wins[d1][d2] += 1.0
    sample['matchup_counts'] = match_counts
    sample['matchup_wins'] = wins
    return sample


def gen_match_wins(field, ev, n_individuals, n_matches_per_ind):
    sample = {}
    sample['deck_counts'] = np.random.multinomial(n_individuals, field)
    sample['match_counts'] = sample['deck_counts'] * n_matches_per_ind
    print(sample['match_counts'])
    print(ev)
    sample['win_counts'] = np.random.binomial(sample['match_counts'], ev)
    sample['loss_counts'] = sample['match_counts'] - sample['win_counts']
    return sample


def gen_winners(field, ev, n_winners, win_count, loss_count):
    """
    P(deck | wins,losses) = P(wins,losses | deck) * P(deck) / P(wins,losses)
                    (w+l choose w) * p(match_win|deck)^wins * p(match_loss|deck)^losses * P(deck)
                  = --------------------------------------------------------------------
                    (w+l choose w) * p(match_win)^wins * p(match_loss)^losses
                              p(match_win|deck)^wins * p(match_loss|deck)^losses
                  = P(deck) * --------------------------------------------------
                              (1/2)^wins * (1/2)^losses
                  = P(deck) * (p(match_win|deck)*2)^wins * ((1-p(match_win|deck))*2)^losses
    """
    p_winner = field * np.power(2*ev, win_count) * np.power(2*(1-ev), loss_count)
    sample = {}
    sample['deck_counts'] = np.random.multinomial(n_winners, p_winner)
    sample['match_counts'] = sample['deck_counts'] * (win_count + loss_count)
    sample['win_counts'] = sample['deck_counts'] * win_count
    sample['loss_counts'] = sample['deck_counts'] * loss_count
    return sample


def gen_league_data(
        session: tf.Session,
        n_archetypes,
        n_rounds,
        n_matches,
        wait_time=10,
        matchup_prior=priors.matchup(),
        ind_decks=0,
        ind_matches=0,
        ind_winners=0,
        winner_win_count = 6,
        winner_loss_count = 2,
        field=None):
    if field is None:
        field = priors.field(n_archetypes).sample()
    p_win, matchups = gen_matchups(n_archetypes, matchup_prior)
    ev = tf.reshape(tf.linalg.matmul(matchups, tf.expand_dims(field, -1)), [-1])
    sample = session.run({'field': field, 'matchups': matchups, 'matchups_free': p_win, 'ev': ev})
    sample['wait_time'] = wait_time
    sim_data = simulation.generate_league(sample['field'], sample['matchups'], n_rounds, n_matches, tries=wait_time)
#    foo = gen_match_wins(sample['field'], sample['ev'], ind_decks, ind_matches)
    matchup_data = gen_matches(sample['field'], sample['matchups'], ind_matches)
    ind_data = gen_winners(sample['field'], sample['ev'], ind_winners, winner_win_count, winner_loss_count)
    obs_data = {'pairing_counts': np.array(sim_data['pairing_counts']),
                'record_counts': np.array(sim_data['record_counts']),
                'n_rounds': n_rounds,
                'n_archetypes': n_archetypes,
                'deck_counts': ind_data['deck_counts'],
                'win_counts': ind_data['win_counts'],
                'loss_counts': ind_data['loss_counts'],
                'matchup_counts': matchup_data['matchup_counts'],
                'matchup_wins': matchup_data['matchup_wins']
                }
    return sample, obs_data


if __name__ == "__main__":
    sess = tf.Session()
#    data = gen_sample_data(sess, 3, 100)
    data = gen_league_data(sess, 3, 3, 100)
    m = np.array([[.5, .7, .2], [.3, .5, .6], [.8, .4, .5]], dtype=np.float32)
    free = np.array([0.4, 0.8, 0.3], dtype=np.float32)
    constructed = sess.run(build_matchup_matrix(tf.constant(free), 3))
    print(constructed)
    deconstructed = sess.run(get_matchup_parameters(tf.constant(m)))
    print(deconstructed)
    assert(np.square(constructed - m).sum() < .0001)
    assert(np.square(deconstructed - free).sum() < .0001)
    indices = sess.run(free_to_matrix(3))
    print(indices)
    assert(indices[0][0] == 2)
    assert(indices[0][1] == 1)
    assert(indices[1][0] == 2)
    assert(indices[1][1] == 0)
    assert(indices[2][0] == 1)
    assert(indices[2][1] == 0)
