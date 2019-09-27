#!/usr/bin/env python

import matplotlib.pyplot as plt
from random import Random
from sys import argv

rand = Random()
match_wins = {}

class League(object):
    def __init__(self, field_distribution, matchups, score_distribution):
        self.matchups = matchups
        self.max_rounds = max(score_distribution.keys())
        self.field = {}
        self.field_distribution = field_distribution
        self.decks = range(len(field_distribution))
        self.n = 0
        self.score_counts = {}
        self.scores = []
        self.score_distribution = score_distribution
        self.records = []
        self.field_record = {}
        self.record_counts = {}
        self.pairings = {}
        for i in range(2*self.max_rounds + 1):
            score = i - self.max_rounds
            self.scores.append(score)
            self.field[score] = {deck: 0 for deck in self.decks}
            self.score_counts[score] = 0
        for s1 in self.scores:
            self.pairings[s1] = {s2: 0 for s2 in self.scores}
        for n in range(self.max_rounds+1):
            for l in range(n+1):
                r = (n-l, l)
                self.records.append(r)
                self.field_record[r] = {deck: 0 for deck in self.decks}
                self.record_counts[r] = 0
        for d in self.decks:
            match_wins[d] = {d2: 0 for d2 in self.decks}

    def current_distribution(self):
        return {d: float(sum([self.field_record[r][d] for r in
            self.records]))/self.n for d in self.decks}

    def add(self, n):
        """Add a single deck, sampled from the field distribution, with a 0-0
        record."""
        new_decks = rand.choices(self.decks, self.field_distribution, k=n)
        for d in new_decks:
            self.field[0][d] = self.field[0][d] + 1
            self.score_counts[0] = self.score_counts[0] + 1
            self.field_record[(0,0)][d] = self.field_record[(0,0)][d] + 1
            self.record_counts[(0,0)] = self.record_counts[(0,0)] + 1
        self.n += n

    def choose_score(self):
        """Sample a score uniformly."""
        p = [self.score_counts[score]/float(self.n) for score in self.scores]
        return rand.choices(self.scores, p, k=1)[0]

    def choose_paired_score(self, score1):
        """Sample the opponent's score given the player's score, according to
        the score pairing distribution."""
        p = [self.score_distribution[score1][score2] for score2 in self.scores]
        return rand.choices(self.scores, p, k=1)[0]

    def choose_deck(self, score=None):
        """Choose a deck, identified by archetype and record, by sampling
        uniformly from those with the exact score (wins minus losses) or
        completely at random if no score is given."""
        if score is None:
            return self.choose_deck(self.choose_score())
        n = float(self.score_counts[score])
        if n > 0:
            options = []
            counts = []
            for (w, l) in self.records:
                if w-l == score:
                    for deck in self.field_record[(w, l)]:
                        k = self.field_record[(w, l)][deck]
                        if k > 0:
                            options.append((deck, (w, l)))
                            counts.append(self.field_record[(w, l)][deck])
            p_record_deck = [k/n for k in counts]
            if len(options) > 0:
                d = rand.choices(options, p_record_deck, k=1)[0]
                if d is not None:
                    return d
        return (None, None)

    def choose_within_one(self, score, tries):
        for i in range(tries):
            (deck, record) = self.choose_deck()
            if record is not None and (record[0] - record[1]) == score:
                return (deck, record, score)
        if self.score_counts.get(score, 0) == 0:
            if self.score_counts.get(score-1, 0) == 0:
                if self.score_counts.get(score+1, 0) == 0:
                    return (None, None, None)
        while(True):
            (deck, record) = self.choose_deck()
            if record is not None:
                other_score = record[0] - record[1]
                if abs(score - other_score) <= 1:
                    return (deck, record, other_score)

    def choose_iterative(self, score, tries):
        max_delta = 2 * self.max_rounds
        for delta in range(max_delta):
            for i in range(tries):
                (deck, record) = self.choose_deck()
                if record is not None:
                    other_score = record[0] - record[1]
                    if abs(other_score - score) <= delta:
                        return deck, record, other_score
        (deck, record) = self.choose_deck()
        if record is not None:
            other_score = record[0] - record[1]
            return deck, record, other_score
        return None, None, None

    def add_record(self, deck, new_w, new_l):
        """Add a deck with a specific record to the record/score counts."""
        new_score = new_w - new_l
        if new_w + new_l > self.max_rounds:
            self.add_record(deck, 0, 0)
        else:
            record = (new_w, new_l)
            self.field[new_score][deck] = self.field[new_score][deck] + 1
            self.score_counts[new_score] = self.score_counts[new_score] + 1
            self.field_record[record][deck] = self.field_record[record][deck] + 1
            self.record_counts[record] = self.record_counts[record] + 1

    def record_pairing(self, s1, s2):
        self.pairings[s1][s2] = self.pairings[s1][s2] + 1

    def play_match(self, tries=None):
        p1_score = self.choose_score()
        (p1_deck, p1_record) = self.choose_deck(p1_score)
        if p1_deck is None:
            return
        if tries is None:
            p2_score = self.choose_paired_score(p1_score)
            (p2_deck, p2_record) = self.choose_deck(p2_score)
        else:
            (p2_deck, p2_record, p2_score) = self.choose_iterative(p1_score, tries)
        if p2_deck is None:
            return
        self.record_pairing(p1_score, p2_score)
        self.field[p1_score][p1_deck] = self.field[p1_score][p1_deck] - 1
        self.field[p2_score][p2_deck] = self.field[p2_score][p2_deck] - 1
        self.score_counts[p1_score] = self.score_counts[p1_score] - 1
        self.score_counts[p2_score] = self.score_counts[p2_score] - 1
        self.field_record[p1_record][p1_deck] = self.field_record[p1_record][p1_deck] - 1
        self.field_record[p2_record][p2_deck] = self.field_record[p2_record][p2_deck] - 1
        self.record_counts[p1_record] = self.record_counts[p1_record] - 1
        self.record_counts[p2_record] = self.record_counts[p2_record] - 1
        p_win_p1 = self.matchups[p1_deck][p2_deck]
        p1_w, p1_l = p1_record
        p2_w, p2_l = p2_record
        if rand.random() < p_win_p1:
            self.add_record(p1_deck, p1_w+1, p1_l)
            self.add_record(p2_deck, p2_w, p2_l+1)
            match_wins[p1_deck][p2_deck] = match_wins[p1_deck][p2_deck] + 1
        else:
            self.add_record(p1_deck, p1_w, p1_l+1)
            self.add_record(p2_deck, p2_w+1, p2_l)
            match_wins[p2_deck][p1_deck] = match_wins[p2_deck][p1_deck] + 1

    def ev(self, deck, score):
        p = 0.0
        for score2 in self.scores:
            p_pair_score = self.score_distribution[score][score2]
            n = self.score_counts[score2]
            if n > 0:
                for deck2 in self.decks:
                    p_pair_deck = self.field[score2][deck2] / float(n)
                    p_win = self.matchups[deck][deck2]
                    p += p_pair_score * p_pair_deck * p_win
        return p


def generate_league(field, matchups, n_rounds, n_samples, n_decks=1000, tries=10):
    n_matches = 10
    print("Generating test data: {} archetypes, {} decks in field, approx {}"
            " pairings per score".format(len(field), n_decks, n_samples))
    scores = [i-n_rounds for i in range((n_rounds*2)+1)]
    score_map = {s1: {s2: 0.0 for s2 in scores} for s1 in scores}
    for i in scores:
        score_map[i][i] = 1.0
    l = League(field, matchups, score_map)
    l.add(n_decks)
    actual_field = l.field[0]
    actual_distribution = {d: actual_field[d]/float(n_decks) for d in actual_field}
    ev = {d: l.ev(d, 0) for d in actual_field}
    counts = {s: [0]*len(field) for s in scores}
    record_counts = {r: [0]*len(field) for r in l.records}
    pairing_counts = {r: [0]*len(field) for r in l.records}
    data_points = 0
    for _i in range(n_samples):
        for j in range(n_matches):
            l.play_match(tries)
    for _i in range(n_samples):
        for j in range(n_matches):
            l.play_match(tries)
        s = l.choose_score()
        (i, r) = l.choose_deck(s)
        if i is not None:
            data_points += 1
            counts[s][i] += 1
            record_counts[r][i] += 1
        s2 = l.choose_paired_score(l.choose_score())
        (i2, r2) = l.choose_deck(s2)
        if i2 is not None:
            data_points += 1
            pairing_counts[r2][i2] += 1
    data = {
            'n_archetypes': len(field),
            'n_rounds': n_rounds,
            'pairings': [counts[s] for s in scores],
            'record_counts': [record_counts[r] for r in l.records],
            'pairing_counts': [pairing_counts[r] for r in l.records],
            'paired_scores': [[0]*len(scores) for s in scores],
            'decks': [[0]*len(field) for s in l.records]
    }
    for s in scores:
        n_score = float(sum(counts[s]))
        p_score = [x/n_score if x > 0 else 0 for x in counts[s]]
        print("\tscore distribution[{}]: {}".format(s, p_score))
    for r in l.records:
        n_record = float(sum(record_counts[r]))
        p_record = [x/n_record if x > 0 else 0 for x in record_counts[r]]
        print("\trecord distribution[{}]: {}".format(r, p_record))
    for d1 in l.decks:
        record = []
        for d2 in l.decks:
            mw = match_wins[d1][d2]
            ml = match_wins[d2][d1]
            p = mw / float(mw+ml) if mw+ml > 0 else 0.0
            record.append(p)
        print("\tEmpirical match wins[{}]: {}".format(d1, record))
    n_wins = {d: sum([match_wins[d][x] for x in actual_field]) for d in actual_field}
    n_losses = {d: sum([match_wins[x][d] for x in actual_field]) for d in actual_field}
    win_rate = {d: n_wins[d]/float(n_wins[d]+n_losses[d]) if n_wins[d] > 0
            else 0.0 for d in actual_field}
    print("\tSample distribution:", actual_distribution)
    print("\tSample EV:", ev)
    print("\tSample empirical wins:", win_rate)
    for s1 in l.scores:
        total = float(sum([l.pairings[s1][s2] for s2 in l.scores]))
        for s2 in l.scores:
            p = l.pairings[s1][s2] / total
            if p > 0:
                print("\tGiven score {}, paired against {}: {}".format(s1, s2, p))
    for s in l.scores:
        p = float(l.score_counts[s]) / l.n
        print("score[{}]: {}".format(s, p))
    print("{} total data points.".format(data_points))
    return data


def generate_league_sample(n_rounds=4, n_samples=100, n_decks=1000, wait_time=10):
    m = [0.3, 0.8, 0.3]
    #m = [0.45, 0.55, 0.45]
    field = [0.3, 0.5, 0.2]
    matchups = {
        0: {0:    0.5,  1:   m[0],  2: m[1]},
        1: {0: 1-m[0],  1:    0.5,  2: m[2]},
        2: {0: 1-m[1],  1: 1-m[2],  2:  0.5}
    }
    return generate_league(field, matchups, n_rounds, n_samples, n_decks, wait_time)


if __name__ == "__main__":
    data = generate_league_sample()
