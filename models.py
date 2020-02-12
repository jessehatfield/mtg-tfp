import abc
import numpy as np
import six
import tensorflow as tf
import tensorflow_probability as tfp
from math import exp
from typing import List
import scipy.stats

import generate
import priors


@six.add_metaclass(abc.ABCMeta)
class UnknownVariable(object):
    @abc.abstractmethod
    def prior(self) -> tfp.distributions.Distribution:
        """Returns a prior distribution over this variable."""
        pass

    @abc.abstractmethod
    def initial_state(self, n=1):
        """Returns an initializer appropriate for this variable."""
        pass

    @abc.abstractstaticmethod
    def bijector(self, n=1) -> tfp.bijectors.Bijector:
        """Returns a tfp.Bijector that can transform this variable from/to an unconstrained space."""
        pass


class UnknownField(UnknownVariable):
    def __init__(self, n_archetypes: int):
        self.n_archetypes = n_archetypes

    def prior(self):
        return priors.field(self.n_archetypes)

    def initial_state(self, n=1):
        return self.prior().sample(n)

    def bijector(self, n=1):
        return tfp.bijectors.SoftmaxCentered()


class UnknownMatchups(UnknownVariable):
    def __init__(self, n_archetypes: int, alpha: float = None):
        self.n_archetypes = n_archetypes
        self.n_params = int(n_archetypes * (n_archetypes-1) / 2)
        self.alpha = alpha

    def prior(self):
        return priors.matchup(self.alpha) if self.alpha else priors.matchup()

    def initial_state(self, n=1):
#        return tfp.distributions.Uniform(0, 1).sample((n, self.n_params))
        return self.prior().sample((n, self.n_params))

    def bijector(self, n=1):
        return tfp.bijectors.Sigmoid()


class UnknownWaitTime(UnknownVariable):
    def __init__(self, rate: float = 1/20.0):
        self.rate = rate

    def prior(self):
        return tfp.distributions.Exponential(self.rate)

    def initial_state(self, n=1):
        return self.prior().sample((n, 1))

    def bijector(self, n=1):
        return tfp.bijectors.Exp()


class TournamentModel:
    """Models the unknown variables in a specific tournament and their interactions.

    Maintains the set of unknown parameters, and provides methods to create necessary components for inference:
    initialization, bijectors, log probability function, etc."""
    def __init__(self, n_archetypes: int):
        self.n_archetypes = n_archetypes
        self.unknown: List[UnknownVariable] = [UnknownField(n_archetypes), UnknownMatchups(n_archetypes)]
        self.vars = {}

    def initial_state(self, n=1):
        return [x.initial_state(n) for x in self.unknown]

    def bijectors(self, n):
        return [x.bijector(n) for x in self.unknown]

    def add_vars(self, var_list, kernel_results):
        self.vars["field"] = var_list[0]
        self.vars["matchups_free"] = var_list[1]
        self.vars["matchup_matrix"] = generate.build_matchup_matrix(
            self.vars["matchups_free"], self.n_archetypes)
        self.vars["ev"] = tf.reshape(
            tf.linalg.matmul(
                self.vars["matchup_matrix"],
                tf.expand_dims(self.vars["field"], -1)),
            self.vars["field"].shape)
        self.vars["kernel_results"] = kernel_results

    def log_prob_fn(self, match_counts, match_wins):
        n_matches = np.sum(match_counts)

        def log_prob(*args):
            return joint_log_prob(self.unknown[0].prior(), self.unknown[1].prior(),
                                  n_matches, match_counts, match_wins,
                                  *args)
        return log_prob


class LeagueModel(TournamentModel):
    def __init__(self, n_archetypes: int, n_rounds: int):
        self.n_archetypes = n_archetypes
        self.n_rounds = int(n_rounds)
        self.n_scores = int((2 * n_rounds) + 1)
        self.n_records = int(((self.n_rounds + 1) * (self.n_rounds + 2)) / 2)
        self.unknown: List[UnknownVariable] = [
            UnknownField(n_archetypes),
            UnknownMatchups(n_archetypes, 12.0),
            UnknownWaitTime()
        ]
        self.vars = {}
        self.p_score = np.zeros(self.n_scores)
        for w in range(self.n_rounds+1):
            for l in range(self.n_rounds+1-w):
                self.p_score[self._score_index(w-l)] += self._record_prior(w, l)
        p_find_base_arr = np.zeros((self.n_scores, self.n_scores))
        for delta in range(self.n_scores):
            for i in range(self.n_scores):
                p = 0.0
                for j in range(i-delta, i+delta+1):
                    if 0 <= j < self.n_scores:
                        p += self.p_score[j]
                p_find_base_arr[i, delta] = p
        self.p_find_base = tf.constant(p_find_base_arr, dtype=tf.float32)

    def add_vars(self, var_list, kernel_results):
        self.vars["field"] = var_list[0]
        self.vars["matchups_free"] = var_list[1]
        self.vars["wait_time"] = var_list[2]
        self.vars["matchup_matrix"] = generate.build_matchup_matrix(
            self.vars["matchups_free"], self.n_archetypes)
        self.vars["score_matrix"] = generate.build_score_matrix(self.vars["wait_time"], self.p_find_base,
                                                                self.p_score, self.n_rounds)
        self.vars["ev"] = tf.reshape(
            tf.linalg.matmul(
                self.vars["matchup_matrix"],
                tf.expand_dims(self.vars["field"], -1)),
            self.vars["field"].shape)
        self.vars["kernel_results"] = kernel_results

    @staticmethod
    def _record_index(wins, losses):
        """Convert a (win, loss) record into a 1D index."""
        n = wins + losses
        return int((n * (n + 1) / 2) + losses)

    def _score_index(self, score):
        """Convert a (win, loss) record into a 1D index."""
        return score + self.n_rounds

    @staticmethod
    def record_prior(n_rounds, w, l):
        n = w + l
        p_n = 1.0 / (n_rounds + 1.0)
        p_record_given_n = scipy.stats.binom.pmf(w, n, 0.5)
        return p_n * p_record_given_n

    def _record_prior(self, w, l):
        return LeagueModel.record_prior(self.n_rounds, w, l)

    def score_matrix_k(self, delta):
        """Compute P(opp.score | pl.score, max score difference)"""
        p_score_score_k = np.zeros((self.n_scores, self.n_scores), dtype=np.float32)
        for j in range(self.n_scores):
            # P(opp.score=s_i | pl.score=s_j, delta)
            #   = P(opp.score=s_i) / sum[d=-delta...delta]{ P(opp.score=s_j+d) }
            #   = P(s_i) / sum[d=-delta...delta]{ P(s_j+d) }
            valid = [i for i in range(j-delta, j+delta+1) if 0 <= i < self.n_scores]
            total = sum([self.p_score[i] for i in valid])
            for i in valid:
                p_score_score_k[i, j] = self.p_score[i] / total
        return p_score_score_k

    def find_opponent_matrix(self, wait_time):
        p_find = np.zeros((self.n_scores, self.n_scores))
        for delta in range(self.n_scores):
            for i in range(self.n_scores):
                p = 0.0
                for j in range(i-delta, i+delta+1):
                    if 0 <= j < self.n_scores:
                        p += self.p_score[j]
                p_find[i, delta] = 1 - exp(-p / wait_time)
        return p_find

    def score_matrix(self, wait_time):
        # P(success|d) = P(wait for next valid deck <= wait_time) = ExponentialCDF(x=wait_time, rate=P(valid))
        # P(success|s1,d) = 1 - exp(-wait_time * sum[s2 within s1 +- d]{P(s2)})
#        p_success = 1 - tf.exp(-self.p_find_base / wait_time)
        p_success_given_delta_score = 1 - tf.exp(tf.scalar_mul(wait_time, -self.p_find_base))
        p_score_given_score = np.zeros((self.n_scores, self.n_scores))
        p_deltas = []
        for delta in range(0, self.n_scores):
            if len(p_deltas) == 0:
                # P(D=0) = P(success|D=0)
                p_delta_given_score = p_success_given_delta_score[:, delta]
            else:
                # P(D=1) = (1-P(D=0)) * P(success|D=1)
                # P(D=2) = (1-P(D=0)-P(D=1)) * P(success|D=2)
                p_delta_given_score = tf.multiply(1.0 - tf.add_n(p_deltas), p_success_given_delta_score[:, delta])
            p_deltas.append(p_delta_given_score)
            # P(S1 | S2) = sum[D]{P(S1 | S2, D) * P(D | S2)}
            p_score_given_score_delta = self.score_matrix_k(delta)
            p_score_given_score += tf.matmul(p_score_given_score_delta, tf.diag(p_delta_given_score))
        return p_score_given_score
#        return tf.eye(self.n_scores, self.n_scores)

    def log_prob_fn(self, pairing_counts, record_counts, deck_counts, win_counts, loss_counts,
                    matchup_counts, matchup_wins):
        """Args:
            pairing_counts: r x d tensor, where each row is the observed distribution of opposing
                            decks given a particular player record.
            record_counts: r x d tensor, where each row is the observed distribution of decks given
                           that deck's player's record.
            deck_counts: rank 1 tensor, where each entry is the number of independent observations of
                         the deck outside of the league context.
            win_counts: rank 1 tensor, where each entry is the total number of wins across independent
                        observations of the deck outside of the league context, with unknown opponents.
            loss_counts: rank 1 tensor, where each entry is the total number of losses across independent
                         observations of the deck outside of the league context, with unknown opponents.
            matchup_counts: d x d tensor, where each entry is the total number of independent matchup
                            observations of a pair of decks outside the league context, with possibly
                            unrelated sampling distribution
            matchup_wins: d x d tensor, where each entry is the number of those independent matchup
                            observations where the first deck won.
        """
        def log_prob(candidate_field, candidate_matchups, candidate_wait_time):
            """Joint log probability for a candidate field distribution, matchup distribution, and wait time"""
            n_hypotheses = candidate_field.shape[0]
            # Priors
            ll_field = self.unknown[0].prior().log_prob(candidate_field)
            ll_matchups = tf.reduce_sum(self.unknown[1].prior().log_prob(candidate_matchups), axis=1)
            ll_wait = tf.reduce_sum(self.unknown[2].prior().log_prob(candidate_wait_time), axis=1)
            # Compute the full matchup matrix from the matchup parameters
            candidate_matchup_matrix = generate.build_matchup_matrix(candidate_matchups, self.n_archetypes)
#            candidate_score_matrix = self.score_matrix(candidate_wait_time)
            score_matrix_full = generate.build_score_matrix(
                tf.reshape(candidate_wait_time, [n_hypotheses, -1]),
                self.p_find_base, self.p_score, self.n_rounds)
            candidate_score_matrix = tf.reshape(score_matrix_full, (n_hypotheses, self.n_scores, self.n_scores))
            # Approximate the distribution of decks at each record and paired against each record
            p_deck_given_record_approx, p_opp_deck_given_pl_record = approximate_p_deck(
                self.n_rounds, self.n_archetypes, candidate_field, candidate_matchup_matrix, candidate_score_matrix)
            # Construct the probability of observed outcomes according to derived parameters:
            ll_priors = ll_field + ll_matchups + ll_wait
            ll_pairings = pairings_log_prob(pairing_counts, p_opp_deck_given_pl_record)
            ll_records = pairings_log_prob(record_counts, p_deck_given_record_approx)
            # To add in independent record observations, calculate per-match EV (dimensions c * d * 1)
            candidate_ev = tf.linalg.matmul(candidate_matchup_matrix, tf.expand_dims(candidate_field, -1))
            # Calculate logs of match results and deck probability
            log_p_win = tf.reshape(tf.log(candidate_ev), (n_hypotheses, self.n_archetypes))  # c * d
            log_p_lose = tf.reshape(tf.log(1.0 - candidate_ev), (n_hypotheses, self.n_archetypes))  # c * d
            log_p_deck = tf.log(candidate_field)  # c * d
            # To add in independent matchup observations, calculate per-pairing log prob (dimensions c * d * d)
            ind_matchups = tfp.distributions.Binomial(probs=candidate_matchup_matrix, total_count=matchup_counts)
            ll_ind_pairings = ind_matchups.log_prob(matchup_wins)
            ll_ind_match = tf.reduce_sum(tf.reduce_sum(ll_ind_pairings, axis=-1), axis=-1)
            # Multiply observed individual results (d) elementwise by log probabilities of results per archetype (c * d)
            # Then sum over archetypes
            ll_obs_wins = log_p_win * win_counts
            ll_obs_losses = log_p_lose * loss_counts
            ll_obs_counts = log_p_deck * deck_counts
            ll_independent = tf.reduce_sum(ll_obs_wins + ll_obs_losses + ll_obs_counts, axis=-1)
            ll = ll_pairings + ll_records + ll_priors + ll_independent + ll_ind_match
            return ll

        return log_prob


def expand_broadcast(tensor: tf.Tensor, n):
    expanded = tf.expand_dims(tensor, 0)
    shape = [dim.value for dim in expanded.shape]
    shape[0] = n
    return tf.broadcast_to(tensor, shape)


def approximate_p_deck(n_rounds, n_archetypes, candidate_field, candidate_matchup_matrix, candidate_score_matrix):
    """Build a tensor that approximates the probability distribution over archetypes at each possible record.

    Args:
        n_rounds: int, total number of rounds, henceforth r
        n_archetypes: int, total number of archetypes, henceforth k
        candidate_field: Tensor of shape (c*k) corresponding to c hypotheses for overall deck distribution
        candidate_matchup_matrix: Tensor of shape (c*k*k) where entry c,i,j is the win percentage of i vs. j
            according to hypothesis c
        candidate_score_matrix: Tensor of shape (c*s*s) where s=2r+1 and entry c,i,j is the probability of
            having score i according to hypothesis c given that the player is paired against someone with score j

    Returns:
        (p_deck_given_record, p_opp_deck_given_pl_record): two c*k*r tensors where r is the number of records, where
        the first gives the distribution of decks *having* each record, and the second gives the distribution of
        decks a player is expected to be *paired against* when the player has each record.
    """
    n_hypotheses = candidate_matchup_matrix.shape[0]
    n_records = int(((n_rounds + 1) * (n_rounds + 2)) / 2)
    n_scores = int((2 * n_rounds) + 1)
    p_deck_init = tf.broadcast_to(tf.expand_dims(candidate_field, -1), (n_hypotheses, n_archetypes, n_records))
    # Calculate based on number of rounds:
    record_arr = np.zeros(n_records)
    transition_win_arr = np.zeros((n_records, n_records))
    transition_lose_arr = np.zeros((n_records, n_records))
    score_given_record_arr = np.zeros((n_scores, n_records))
    record_given_score_arr = np.zeros((n_records, n_scores))
    for w1 in range(n_rounds + 1):
        for l1 in range(n_rounds + 1 - w1):
            score_index = n_rounds + w1 - l1
            record_index = LeagueModel._record_index(w1, l1)
            score_given_record_arr[score_index, record_index] = 1.0
            record_given_score_arr[record_index, score_index] = LeagueModel.record_prior(n_rounds, w1, l1)
            record_arr[record_index] = LeagueModel.record_prior(n_rounds, w1, l1)
            for w2 in range(n_rounds + 1):
                for l2 in range(n_rounds + 1 - w2):
                    record_index2 = LeagueModel._record_index(w2, l2)
                    # Fill in P(w,l | w2,l2,win) and P(w,l | w2,l2,lose)
                    if (w2 + l2) == n_rounds and (w1 + l1) == 0:
                        # Reset to zero after maximum number of matches, regardless of result
                        transition_win_arr[record_index, record_index2] = 1
                        transition_lose_arr[record_index, record_index2] = 1
                    elif l1 == l2 and w1 == w2 + 1:
                        transition_win_arr[record_index, record_index2] = 1
                    elif w1 == w2 and l1 == l2 + 1:
                        transition_lose_arr[record_index, record_index2] = 1

    p_record = expand_broadcast(tf.constant(record_arr, dtype=tf.float32), n_hypotheses)
    p_record_given_record_and_win = expand_broadcast(tf.constant(transition_win_arr, dtype=tf.float32), n_hypotheses)
    p_record_given_record_and_lose = expand_broadcast(tf.constant(transition_lose_arr, dtype=tf.float32), n_hypotheses)
    p_score_given_record = expand_broadcast(tf.constant(score_given_record_arr, dtype=tf.float32), n_hypotheses)
    # Normalize columns to sum to one
    record_given_score_arr = record_given_score_arr / record_given_score_arr.sum(axis=0)
    p_record_given_score = expand_broadcast(tf.constant(record_given_score_arr, dtype=tf.float32), n_hypotheses)

    p_opp_rec_given_pl_score = tf.matmul(p_record_given_score, candidate_score_matrix)
    p_opp_record_given_pl_record = tf.matmul(p_opp_rec_given_pl_score, p_score_given_record)

    # Define a loop to iteratively approximate p_deck_given_record
#    def update_p_deck(_, p_deck_given_record, p_win_given_decks, p_opp_score_given_pl_score, _):
#    def update_p_deck(_, p_deck_given_record, p_win_given_decks):
    def update_p_deck(_, p_deck_given_record):
        # Derive from other distributions:
#        p_opp_rec_given_pl_score = tf.matmul(p_record_given_score, p_opp_score_given_pl_score)
#        p_opp_record_given_pl_record = tf.matmul(p_opp_rec_given_pl_score, p_score_given_record)
        p_joint_records = tf.matmul(p_opp_record_given_pl_record, tf.linalg.diag(p_record))
        p_opp_deck_and_pl_record = tf.matmul(p_deck_given_record, p_joint_records)
        p_record_and_win_given_deck = tf.matmul(candidate_matchup_matrix, p_opp_deck_and_pl_record)
        p_record_and_lose_given_deck = tf.matmul(candidate_matchup_matrix, p_opp_deck_and_pl_record, transpose_a=True)
#        p_record_and_win_given_deck = tf.matmul(p_win_given_decks, p_opp_deck_and_pl_record)
#        p_record_and_lose_given_deck = tf.matmul(p_win_given_decks, p_opp_deck_and_pl_record, transpose_a=True)
        #            p_deck_record_and_win = tf.multiply(p_record_and_win_given_deck, p_deck_given_record)
        #            p_deck_record_and_lose = tf.multiply(p_record_and_lose_given_deck, p_deck_given_record)
        p_deck = tf.reshape(tf.matmul(p_deck_given_record, tf.expand_dims(p_record, -1)), [n_hypotheses, -1])
        p_deck_diag = tf.linalg.diag(p_deck)
        p_deck_record_and_win = tf.matmul(p_deck_diag, p_record_and_win_given_deck)
        p_deck_record_and_lose = tf.matmul(p_deck_diag, p_record_and_lose_given_deck)
        p_deck_and_won_and_next_record = tf.matmul(p_deck_record_and_win, p_record_given_record_and_win,
                                                   transpose_b=True)
        p_deck_and_lost_and_next_record = tf.matmul(p_deck_record_and_lose, p_record_given_record_and_lose,
                                                    transpose_b=True)
        p_deck_and_next_record = p_deck_and_won_and_next_record + p_deck_and_lost_and_next_record
        p_record_inv_diag = tf.linalg.diag(1.0 / p_record)
        p_deck_given_next_record = tf.matmul(p_deck_and_next_record, p_record_inv_diag)
        return p_deck_given_record, p_deck_given_next_record
#               p_win_given_decks, p_opp_score_given_pl_score, p_opp_record_given_pl_record
#               p_win_given_decks

#    _, p_deck_given_record_approx, _, _, p_opp_record_given_pl_record = tf.while_loop(
    _, p_deck_given_record_approx = tf.while_loop(
#        lambda old, new, m, s, r: tf.less(.0001, tf.reduce_sum(tf.abs(tf.subtract(old, new)))),
        lambda old, new: tf.less(.0001, tf.reduce_sum(tf.abs(tf.subtract(old, new)))),
        update_p_deck,
#        (tf.zeros_like(p_deck_init), p_deck_init, candidate_matchup_matrix, candidate_score_matrix),
        (tf.zeros_like(p_deck_init), p_deck_init),
        parallel_iterations=1, maximum_iterations=10)
    p_opp_deck_given_pl_record = tf.matmul(p_deck_given_record_approx, p_opp_record_given_pl_record)
    return p_deck_given_record_approx, p_opp_deck_given_pl_record


def pairings_log_prob(pairing_counts, candidate_p_deck_given_record):
    """
    param pairing_counts: r x d tensor, where each row is the observed distribution of pairings for a particular record
    param candidate_p_deck_given_record: d x r tensor, where each entry gives the probability of a deck having
            archetype i given that it is associated with record j"""
#    # Flatten the table of record/deck pairing counts ??
#    flattened_pairing_counts = tf.reshape(pairing_counts, [-1])
#    # And sum over the different decks to get the per-record pairing totals
#    record_totals = pairing_counts.sum(axis=1)
    record_totals = tf.reduce_sum(pairing_counts, axis=1)
    rv_pairing_counts = generate.rv_pairing_counts(record_totals, candidate_p_deck_given_record)
    return tf.reduce_sum(rv_pairing_counts.log_prob(pairing_counts), axis=1)


def joint_log_prob(
        field_prior: tfp.distributions.Distribution,
        matchup_prior: tfp.distributions.Distribution,
        n_matches,
        obs_match_counts,
        obs_match_wins,
        candidate_field,
        candidate_matchups):
    """Joint log probability for a candidate field and matchup distribution"""
    # Priors
    ll_field = field_prior.log_prob(candidate_field)
    # Matchup matrix has n*n entries but only n*(n-1)/2 independent parameters,
    # because M[i,j]=1-M[j,i] and therefore M[i,i]=0.5. Therefore, only compute
    # log likelihood over entries where j>i.
    n = obs_match_counts.shape[0]
    #    matchup_free = fill_triangular_inverse(
    #        tf.slice(candidate_matchups, [0,1], [n-1, n-1]), upper=True)
    ll_matchups = tf.reduce_sum(matchup_prior.log_prob(candidate_matchups))
    # Probability of match counts given field distribution
    # (observed counts should be a matrix, but the distribution is defined over
    # a flat vector, so first we flatten the data)
    flattened_match_counts = tf.reshape(obs_match_counts, [-1])
    rv_match_counts = generate.rv_match_counts(candidate_field, n_matches)
    ll_counts = tf.reduce_sum(rv_match_counts.log_prob(flattened_match_counts))
    # Probability of outcomes given match counts and matchups
    # (observed data is number of wins, but the distribution is defined over
    # combination of wins and losses, which we can derive from wins and counts)
    obs_outcome = generate.wins_to_outcomes(obs_match_wins, obs_match_counts)
    candidate_matchup_matrix = generate.build_matchup_matrix(candidate_matchups, n)
    rv_match_outcomes = generate.rv_outcomes(obs_match_counts, candidate_matchup_matrix)
    ll_wins = tf.reduce_sum(rv_match_outcomes.log_prob(obs_outcome))
    return ll_field + ll_matchups + ll_counts + ll_wins
