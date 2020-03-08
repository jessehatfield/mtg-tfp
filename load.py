#!/usr/bin/env python

from matches import process_results

import pickle
import sys
import tensorflow as tf

def load_results(in_dir, plot_dir=None):
    with open(in_dir + '/results.pkl', 'rb') as in_file:
        results = pickle.load(in_file)
    with open(in_dir + '/labels.pkl', 'rb') as in_file:
        labels = pickle.load(in_file)
    process_results(tf.Session(), results, labels,
                    results['burn_in'] if 'burn_in' in results else 0,
                    plot_dir=plot_dir, show_plots=plot_dir is None)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        load_results(sys.argv[1])
    elif len(sys.argv) == 3:
        load_results(sys.argv[1], sys.argv[2])
