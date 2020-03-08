#!/usr/bin/env python

import tensorflow as tf
import tensorflow_probability as tfp

from math import ceil
import numpy as np

import argparse
import csv
import pickle
import sys

import generate
import plots
from models import LeagueModel


def run_league_inference(
        session,
        data,
        num_samples=1000,
        burn_in=100,
        leapfrog_steps=1,
        sample_interval=0,
        step_size=0.25,
        num_chains=4):
    print("----Running MCMC----")
    league_model = LeagueModel(data['n_archetypes'], data['n_rounds'])
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        step_size = tf.get_variable(
            name="step_size",
            initializer=tf.constant(step_size, dtype=tf.float32),
            trainable=False,
            use_resource=True)
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=league_model.log_prob_fn(data['pairing_counts'], data['record_counts'],
                                                    data['deck_counts'], data['win_counts'],
                                                    data['loss_counts'], data['matchup_counts'],
                                                    data['matchup_wins']),
        num_leapfrog_steps=leapfrog_steps,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=int(burn_in*.8)),
        state_gradients_are_stopped=True)
    hmc = tfp.mcmc.TransformedTransitionKernel(inner_kernel, league_model.bijectors(num_chains))
    sample_vars, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_steps_between_results=sample_interval,
        num_burnin_steps=burn_in,
        current_state=league_model.initial_state(num_chains),
        kernel=hmc)
    league_model.add_vars(sample_vars, kernel_results)
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    session.run([init_g])
    session.run([init_l])
    results = session.run(league_model.vars)
    return league_model, results


def indent(obj, prefix="\t"):
    return prefix + str(obj).replace("\n", "\n{}".format(prefix))


def summarize_run(mcmc_results, archetype_labels, burn_in):
    print("Acceptance rate: {}".format(mcmc_results["kernel_results"].inner_results.is_accepted.mean()))
    print("Final step size: {}".format(mcmc_results["kernel_results"]
                                       .inner_results.extra.step_size_assign[-100:].mean()))
    avg_field = np.mean(mcmc_results["field"][burn_in:, :, :], axis=0)
    avg_matchups = np.mean(mcmc_results["matchup_matrix"][burn_in:, :, :, :], axis=0)
    avg_wait = np.mean(mcmc_results["wait_time"][burn_in:, :, :], axis=0)
    avg_score_matrix = np.mean(mcmc_results["score_matrix"], axis=0)
    print("M == \n{}\nfield==\n{}\nwait time==\n{}\nscore_matrix=\n{}".format(
        indent(avg_matchups), indent(avg_field), indent(avg_wait), indent(avg_score_matrix)))
    print("Average score pairings:")
    r = (len(avg_score_matrix) - 1) / 2
    for j in range(len(avg_score_matrix)):
        for i in range(len(avg_score_matrix)):
            print("\tP(opp.score={} | pl.score={}) == {}%".format(i-r, j-r, 100*avg_score_matrix[i][j]))
    print("Field distribution stats:")
    flat_field = plots.reduce_dimension(mcmc_results["field"][burn_in:, :, :], 2)
    field_summary = np.vstack([np.mean(flat_field, axis=0, keepdims=True),
                               np.std(flat_field, axis=0, keepdims=True),
                               np.nanpercentile(flat_field, [5, 25, 50, 75, 95], axis=0)]).transpose()
    print(f"\t{'Archetype':20}{'Mean':>10}{'Std.Dev':>10}{'5%':>10}{'25%':>10}{'50%':>10}{'75%':>10}{'95%':>10}")
    for i in range(len(archetype_labels)):
        stats = ("{:10f}" * len(field_summary[i])).format(*field_summary[i])
        print(f"\t{archetype_labels[i]:20}{stats}")
    flat_matchups = plots.reduce_dimension(mcmc_results["matchup_matrix"][burn_in:, :, :, :], 3)
    fixed_ev = np.matmul(flat_matchups, np.expand_dims(flat_field, -1))[:, :, 0]
    ev_summary = np.vstack([np.mean(fixed_ev, axis=0, keepdims=True),
                            np.std(fixed_ev, axis=0, keepdims=True),
                            np.nanpercentile(fixed_ev, [5, 25, 50, 75, 95], axis=0)]).transpose()
    print("EV stats:")
    print(f"\t{'Archetype':20}{'Mean':>10}{'Std.Dev':>10}{'5%':>10}{'25%':>10}{'50%':>10}{'75%':>10}{'95%':>10}")
    for i in range(len(archetype_labels)):
        stats = ("{:10f}" * len(ev_summary[i])).format(*ev_summary[i])
        print(f"\t{archetype_labels[i]:20}{stats}")


def plot_traces(mcmc_results, archetype_labels, out_dir=None):
    session = tf.get_default_session()
    if session is None:
        session = tf.Session()
    indices = session.run(generate.free_to_matrix(len(archetype_labels)))
    tlp = mcmc_results["kernel_results"].inner_results.proposed_results.target_log_prob
    plots.trace(tlp,
                title="Log-Likelihood of MCMC Samples",
                xlabel="$t$",
                ylabel="$\\log P(X|\\Theta_t)$",
                filename=None if out_dir is None else f"{out_dir}/ll_mcmc_samples.png")
    plots.trace(mcmc_results["wait_time"], xlabel="$t$", title="Parameter Trace: Wait Time",
                ylabel="$w$",
                filename=None if out_dir is None else f"{out_dir}/trace_wait_time.png")
    plots.trace(mcmc_results["field"], xlabel="$t$", title="Parameter Trace: $P(Deck)$",
                plots=archetype_labels,
                filename=None if out_dir is None else f"{out_dir}/trace_field.png")
    plots.trace(mcmc_results["matchups_free"], xlabel="$t$", title="Parameter Trace: $P(w|d_i vs. d_j)$",
                plots=["{} vs. {}".format(archetype_labels[pair[0]], archetype_labels[pair[1]]) for pair in indices],
                filename=None if out_dir is None else f"{out_dir}/trace_matchups.png")


def plot_posteriors(mcmc_results, archetype_labels, burn_in, out_dir=None):
#    plots.matchup_matrix_posterior(mcmc_results["score_matrix"])
    n = len(mcmc_results["wait_time"])
    interval = int(ceil((n-burn_in) / 500.0))
    plots.constant_posterior(mcmc_results["wait_time"][burn_in::interval, :, :], "Posterior Wait Time",
                             filename=None if out_dir is None else f"{out_dir}/wait_time.png")
    plots.field_posterior(mcmc_results["field"][burn_in::interval, :, :], xlabel="Posterior Field Proportion",
                          dist_labels=archetype_labels,
                          filename=None if out_dir is None else f"{out_dir}/field.png")
    plots.matchup_matrix_posterior(mcmc_results["matchup_matrix"][burn_in::interval, :, :, :],
                                   ev=mcmc_results["ev"][burn_in::interval, :, :],
                                   title="Posterior Matchup Matrix",
                                   archetypes=archetype_labels,
                                   filename=None if out_dir is None else f"{out_dir}/matchups.png")


def evaluate_parameters(session, model, obs_data, field, matchups, wait):
    true_field = tf.expand_dims(tf.constant(field), 0)
    true_matchup_matrix = tf.constant(matchups)
    true_matchups = tf.expand_dims(generate.get_matchup_parameters(true_matchup_matrix), 0)
    true_wait = tf.reshape(tf.constant(wait, dtype=tf.float32), (1, 1))
    ll_true = model.log_prob_fn(
        obs_data['pairing_counts'], obs_data['record_counts'],
        obs_data['deck_counts'], obs_data['win_counts'], obs_data['loss_counts'],
        obs_data['matchup_counts'], obs_data['matchup_wins']
    )(true_field, true_matchups, true_wait)
    ll, matchup_params = session.run([ll_true, true_matchups])
    return ll


def process_results(session, results, labels, burn_in, out_dir=None, show_plots=True, plot_dir=None):
    results['burn_in'] = burn_in
    summarize_run(results, labels, burn_in)
    if out_dir is not None:
        with open(out_dir + "/results.pkl", "wb") as out_file:
            pickle.dump(results, out_file)
        with open(out_dir + "/labels.pkl", "wb") as out_file:
            pickle.dump(labels, out_file)
        field_samples = results['field'][burn_in:, :, :]
        field_2d = np.reshape(field_samples, (-1, len(labels)))
        field_header = ','.join(labels)
        np.savetxt(out_dir + "/trace_field.csv", field_2d, fmt="%10.7f", delimiter=",", header=field_header)
        ev_samples = results['ev'][burn_in:, :, :]
        ev_2d = np.reshape(ev_samples, (-1, len(labels)))
        np.savetxt(out_dir + "/trace_ev.csv", ev_2d, fmt="%10.7f", delimiter=",", header=field_header)
        matchup_samples = results['matchups_free'][burn_in:, :, :]
        matchups_2d = np.reshape(matchup_samples, (-1, matchup_samples.shape[2]))
        indices = session.run(generate.free_to_matrix(len(labels)))
        matchups_header = ','.join([f"{labels[pair[0]]} vs. {labels[pair[1]]}" for pair in indices])
        np.savetxt(out_dir + "/trace_matchups.csv", matchups_2d, fmt="%10.7f", delimiter=",", header=matchups_header)
    if show_plots:
        plot_traces(results, labels)
        plot_posteriors(results, labels, burn_in)
    if plot_dir is not None:
        plot_traces(results, labels, out_dir=plot_dir)
        plot_posteriors(results, labels, burn_in, out_dir=plot_dir)


def generate_sample_data(session, n_archetypes, n_rounds, wait_time, n_matches, extra_matches):
    gen_params, gen_data = generate.gen_league_data(session, n_archetypes, n_rounds, n_matches, wait_time,
                                                    ind_winners=20, ind_matches=extra_matches)
    print("----Ground Truth----\nM ==\n{}\nfield == {}\nev == {}".format(
        indent(gen_params['matchups']), gen_params['field'], gen_params['ev']))
    print("----Data----\npairings ==\n{}\nrecords ==\n{}\nindependent observations==\n{}\n{}\n{}\n".format(
        indent(gen_data['pairing_counts']), indent(gen_data['record_counts']),
        indent(gen_data['deck_counts']), indent(gen_data['win_counts']), indent(gen_data['loss_counts'])))
    print("independent matchup results==\ncounts:{}\n  wins:{}\n".format(
        indent(gen_data['matchup_counts']), indent(gen_data['matchup_wins'])))
    return gen_params, gen_data, ["$d_{}$".format(i) for i in range(n_archetypes)]


def run_example():
    sess = tf.Session()
    n_chains = 4
    burn_in = 100
    n_samples = 5000
    params, data, labels = generate_sample_data(sess, 3, 3, 10, 100, 200)
    t_model, results = run_league_inference(sess, data, num_samples=n_samples, burn_in=burn_in, num_chains=n_chains)
    true_ll = evaluate_parameters(sess, t_model, data, params['field'], params['matchups'], params['wait_time'])
    print("Ground truth log-likelihood:", true_ll)
    process_results(sess, results, labels, burn_in, "temp")


class InputData(object):
    def __init__(self):
        self.archetype_pairings = {}
        self.archetype_totals = {}
        self.score_pairings = {}
        self.archetype_records = {}
        self.n_rounds = 0
        self.substitutions = {'Rogue': 'Misc.', 'Unknown': 'Misc.'}
        self.deck_counts = {}
        self.win_counts = {}
        self.loss_counts = {}
        self.extra_match_counts = {}
        self.extra_match_wins = {}

    def load_substitutions(self, substitution_file):
        with open(substitution_file) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 2:
                    self.substitutions[row[1].strip()] = row[0].strip()
        self.substitutions['Rogue'] = self.substitutions.get('Rogue', 'Misc.')
        self.substitutions['Unknown'] = self.substitutions.get('Unknown', 'Misc.')

    def load_league_pairings(self, league_data_file):
        with open(league_data_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                w = int(row['w']) if len(row.get('w', '')) > 0 else None
                l = int(row['l']) if len(row.get('l', '')) > 0 else None
                opponent = row.get('Opposing Deck', row.get('deck', ''))
                opponent = self.substitutions.get(opponent, opponent)
                if w is None or l is None or len(opponent) == 0:
                    print('SKIPPING ROW: {}'.format(row))
                    continue
                opp_w = int(row['opp_w']) if len(row.get('opp_w', '')) > 0 else None
                opp_l = int(row['opp_l']) if len(row.get('opp_l', '')) > 0 else None
                score = w-l
                self.n_rounds = max(self.n_rounds, w+l)
                self.archetype_totals[opponent] = self.archetype_totals.get(opponent, 0) + 1
                if opponent not in self.archetype_pairings:
                    self.archetype_pairings[opponent] = {}
                if opponent not in self.archetype_records:
                    self.archetype_records[opponent] = {}
                if opp_w is not None and opp_l is not None:
                    opp_record = (opp_w, opp_l)
                    opp_score = opp_w - opp_l
                    if opp_score not in self.score_pairings:
                        self.score_pairings[opp_score] = {}
                    self.archetype_records[opponent][opp_record] = self.archetype_records[opponent].get(opp_record, 0) + 1
                    self.score_pairings[opp_score][score] = self.score_pairings[opp_score].get(score, 0) + 1
                else:
                    record = (w, l)
                    self.archetype_pairings[opponent][record] = self.archetype_pairings[opponent].get(record, 0) + 1
    def load_final_records(self, record_data_file):
        with open(record_data_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                w = int(row['w']) if len(row.get('w', '')) > 0 else None
                l = int(row['l']) if len(row.get('l', '')) > 0 else None
                deck = row.get('deck', '')
                deck = self.substitutions.get(deck, deck)
                if w is None or l is None or len(deck) == 0:
                    print('SKIPPING ROW: {}'.format(row))
                    continue
                self.archetype_totals[deck] = self.archetype_totals.get(deck, 0)
                self.archetype_pairings[deck] = self.archetype_pairings.get(deck, {})
                self.archetype_records[deck] = self.archetype_records.get(deck, {})
                self.deck_counts[deck] = self.deck_counts.get(deck, 0) + 1
                self.win_counts[deck] = self.win_counts.get(deck, 0) + w
                self.loss_counts[deck] = self.loss_counts.get(deck, 0) + l

    def load_matchup_data(self, matchup_file):
        with open(matchup_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                deck1 = row.get('deck 1', row.get('deck', row.get('archetype', row.get('archetype 1', ''))))
                deck1 = self.substitutions.get(deck1, deck1)
                deck2 = row.get('deck 2', row.get('opp_deck', row.get('opp_archetype', row.get('archetype 2', ''))))
                deck2 = self.substitutions.get(deck2, deck2)
                w = int(row['w']) if len(row.get('w', '')) > 0 else None
                l = int(row['l']) if len(row.get('l', '')) > 0 else None
                n = int(row['total']) if len(row.get('total', '')) > 0 else None
                if w is None or (l is None and n is None):
                    print('SKIPPING ROW: {}'.format(row))
                    continue
                if n is None:
                    n = w + l
                if deck1 == deck2:
                    continue
                elif self.extra_match_counts.get(deck2, {}).get(deck1, 0) == n:
                    print('SKIPPING {} vs. {}, already processed in reverse order'.format(deck2, deck1))
                    continue
                self.extra_match_counts[deck1] = self.extra_match_counts.get(deck1, {})
                self.extra_match_counts[deck1][deck2] = self.extra_match_counts[deck1].get(deck2, 0) + n
                self.extra_match_wins[deck1] = self.extra_match_wins.get(deck1, {})
                self.extra_match_wins[deck1][deck2] = self.extra_match_wins[deck1].get(deck2, 0) + w

    def gather_data(self, selection):
        archetypes = list(self.archetype_totals.keys())
        archetypes.sort(key=lambda x: 1 if x == 'Misc.' else -(self.archetype_totals[x]+self.deck_counts.get(x, 0)))
        if selection.isdigit():
            n = min(int(selection), len(archetypes)-1)
        else:
            archetypes = [selection] + [a for a in archetypes if a != selection]
            n = 1
        pairing_counts = []
        score_counts = []
        record_counts = []
        for i in range((2*self.n_rounds)+1):
            oppscore_distribution = []
            score = i - self.n_rounds
            for j in range((2*self.n_rounds)+1):
                oppscore = j - self.n_rounds
                oppscore_distribution.append(self.score_pairings.get(oppscore, {}).get(score, 0))
            score_counts.append(oppscore_distribution)
        for n_matches in range(self.n_rounds+1):
            for l in range(n_matches+1):
                w = n_matches - l
                deck_distribution = []
                paired_deck_distribution = []
                for deck in archetypes:
                    deck_distribution.append(self.archetype_records[deck].get((w, l), 0))
                    paired_deck_distribution.append(self.archetype_pairings[deck].get((w, l), 0))
                record_counts.append(deck_distribution)
                pairing_counts.append(paired_deck_distribution)
        archetype_counts = [self.archetype_totals[x] for x in archetypes]
        archetype_proportions = [archetype_counts[i] / float(sum(archetype_counts)) for i in range(n)]
        archetype_proportions.append(1.0 - sum(archetype_proportions))
        fully_specified_data = {
            'n_archetypes': len(archetypes),
            'n_rounds': self.n_rounds,
            'paired_scores': score_counts,
            'pairing_counts': pairing_counts,
            'record_counts': record_counts,
            'deck_counts': [self.deck_counts.get(x, 0) for x in archetypes],
            'win_counts': [self.win_counts.get(x, 0) for x in archetypes],
            'loss_counts': [self.loss_counts.get(x, 0) for x in archetypes],
            'matchup_counts': np.array([[self.extra_match_counts.get(x, {}).get(y, 0) for y in archetypes] for x in archetypes], dtype=np.float64),
            'matchup_wins': np.array([[self.extra_match_wins.get(x, {}).get(y, 0) for y in archetypes] for x in archetypes], dtype=np.float64)
        }
        archetype_list = archetypes[:n] + ["Misc."]
        consolidated_data = self.consolidate(fully_specified_data, n)
        consolidated_data['archetypes'] = archetype_list
        consolidated_data['obs_proportion'] = archetype_proportions
        print("Loaded data:")
        for key in consolidated_data:
            print("\t", key, consolidated_data[key])
        return consolidated_data, archetype_list

    def consolidate(self, data, n):
        transformed = {
            'n_archetypes': n+1,
            'n_rounds': data['n_rounds'],
            'paired_scores': data['paired_scores'],
            'pairing_counts': [],
            'record_counts': []
        }
        for r_i in range(len(data['record_counts'])):
            for key in {'pairing_counts', 'record_counts'}:
                new_list = [data[key][r_i][j] for j in range(n)]
                total = sum([data[key][r_i][j] for j in range(n, len(data[key][r_i]))])
                new_list.append(total)
                transformed[key].append(new_list)
            for key in {'deck_counts', 'win_counts', 'loss_counts'}:
                new_list = [data[key][j] for j in range(n)]
                total = sum([data[key][j] for j in range(n, len(data[key]))])
                new_list.append(total)
                transformed[key] = new_list
        for key in {'matchup_counts', 'matchup_wins'}:
            temp_counts = []
            for i in range(len(data[key])):
                old_counts = data[key][i]
                new_counts = [data[key][i][j] for j in range(n)]
                total = sum([data[key][i][j] for j in range(n, len(old_counts))])
                new_counts.append(total)
                temp_counts.append(new_counts)
            transformed_counts = [temp_counts[i] for i in range(n)]
            n_misc = len(data[key])
            misc_counts = [sum([temp_counts[i][j] for i in range(n, n_misc)]) for j in range(n+1)]
            transformed_counts.append(misc_counts)
            transformed[key] = np.array(transformed_counts, dtype=np.float32)
        return transformed


if __name__ == "__main__":
    if len(sys.argv) > -1:
        parser = argparse.ArgumentParser(description='Run Bayesian inference over MTGO data.')
        parser.add_argument('-s', '--substitutions', action='store', nargs='+', default=[],
                            help='CSV file(s) containing name substitutions: each row should be'
                            ' "to,from" where "from" is replaced with "to" before processing.',
                            metavar='filename')
        parser.add_argument('-m', '--matchup-data', action='store', nargs='+', default=[],
                            help='CSV file(s) containing matchup data: expects columns'
                            ' for deck one ("deck 1", "deck", "archetype 1", or "archetype"),'
                            ' deck two ("deck 2", "opp_deck", "archetype 2", or "opp_archetype"),'
                            ' games won by deck one ("w"),'
                            ' and games lost by deck one/won by deck two ("l")',
                            metavar='filename')
        parser.add_argument('-l', '--league-pairings', action='store', nargs='+', default=[],
                            help='CSV file(s) containing MTGO league pairing counts',
                            metavar='filename')
        parser.add_argument('-r', '--final-records', action='store', nargs='+', default=[],
                            help='CSV file(s) containing final tournament records',
                            metavar='filename')
        parser.add_argument('-d', '--decks', action='store', default='9',
                            help='Either the maximum number of individual decks to consider at once;'
                            ' or a single deck name to consider only one discrete deck. In either'
                            ' case, all other decks will be consolidated into a catch-all category.')
        parser.add_argument('-o', '--output-dir', action='store', default=None,
                            help='Directory to which to save the trace of the run.')
        parser.add_argument('-p', '--plot-dir', action='store', default=None,
                            help='Directory to which to save plots.')
        arguments = parser.parse_args(sys.argv[1:])

        input_data = InputData()
        for substitution_file in arguments.substitutions:
            input_data.load_substitutions(substitution_file)
        for league_data_file in arguments.league_pairings:
            input_data.load_league_pairings(league_data_file)
        for record_data_file in arguments.final_records:
            input_data.load_final_records(record_data_file)
        for matchup_data_file in arguments.matchup_data:
            input_data.load_matchup_data(matchup_data_file)

        data, labels = input_data.gather_data(arguments.decks)

        sess = tf.Session()
        n_samples = 21000
        burn_in = 1000
        t_model, results = run_league_inference(sess, data,
                                                num_samples=n_samples, sample_interval=0,
                                                burn_in=burn_in, num_chains=4)
        process_results(sess, results, labels, burn_in, arguments.output_dir, False, plot_dir=arguments.plot_dir)
    else:
        print('Running example...')
        run_example()
