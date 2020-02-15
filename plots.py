import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import math


def fill_interval(ax, x, y, confidence, color, alpha, samples=None):
    left = 0.5 - (confidence / 2.0)
    right = 0.5 + (confidence / 2.0)
    if samples is None:
        c = np.cumsum(y)
        total = c[-1]
        fill = (left * total < c) * (c < right * total)
    else:
        min_value = np.nanpercentile(samples, left * 100)
        max_value = np.nanpercentile(samples, right * 100)
        if min_value == max_value:
            return
        included = (min_value < x) * (max_value >= x)
        fill = np.nonzero(included)
    ax.fill_between(x[fill], y[fill], color=color, alpha=alpha)


def p_posterior(data, ax=None, hist=False, color=None, rug=True, xlabel=None, dist_label=None):
    if min(data) == max(data):
        if ax is None:
            ax = plt.gca()
#        plt.ylim(bottom=0, top=1)
#        ax.axvline(data[0], color='b', linewidth=2)
        return
    ax = sns.distplot(data, ax=ax, color=color, hist=hist, norm_hist=hist, rug=rug,
                      axlabel=False if xlabel is None else xlabel,
                      label=dist_label)
    plt.ylim(bottom=0)
    x, y = ax.get_lines()[-1].get_data()
    color = ax.get_lines()[-1].get_color()
    fill_interval(ax, x, y, .99, color, .2, data)
    fill_interval(ax, x, y, .95, color, .2, data)
    fill_interval(ax, x, y, .5, color, .2, data)
    ax.axvline(0.5, c='#888888', linestyle='--', linewidth=1)
    return x, y


def matchup_matrix_posterior(samples, ev=None, title=None, archetypes=None, filename=None):
    n_archetypes = samples.shape[-1]
    samples = reduce_dimension(samples, 3)
    if ev is not None:
        ev = reduce_dimension(ev, 2)
    k = 1
    ax = plt.gca()
    plt.xlim([0, 1])
    for i in range(n_archetypes):
        for j in range(n_archetypes):
            sub = plt.subplot(n_archetypes, n_archetypes, k, sharex=ax)
            if i != j or ev is None:
                p_posterior(samples[:, i, j], ax=sub)
            elif ev is not None:
                p_posterior(ev[:, i], ax=sub, color='#009922')
            if archetypes is not None:
                if i == 0:
                    sub.set_title(archetypes[j])
                if j == 0:
                    sub.set_ylabel(archetypes[i], size='large')
            k += 1
    if title is not None:
        plt.suptitle(title)
    if filename is None:
        plt.show()
    else:
        plt.gcf().set_size_inches(2 + 1.5*n_archetypes, 1 + 1.5*n_archetypes)
        plt.savefig(filename)
        plt.close()


def reduce_dimension(arr, ndims):
    if len(arr) > ndims:
        product = 1
        for dim_size in arr.shape[:(1-ndims)]:
            product *= dim_size
        new_shape = [product] + list(arr.shape[(1-ndims):])
        return np.reshape(arr, new_shape)


def field_posterior(samples, xlabel=None, dist_labels=None, filename=None):
    samples = reduce_dimension(samples, 2)
    n_archetypes = samples.shape[1]
    ax = plt.gca()
    plt.xlim([0, 1])
    ymax = 0.01
    for i in range(n_archetypes):
        dist_label = None if dist_labels is None else dist_labels[i]
        x, y = p_posterior(samples[:, i], ax=ax, xlabel=xlabel, dist_label=dist_label)
        ymax = max(ymax, max(y))
    plt.ylim([0, ymax*1.2])
    if filename is None:
        plt.show()
    else:
        plt.gcf().set_size_inches(16, 8)
        plt.savefig(filename)
        plt.close()


def constant_posterior(samples, label=None, filename=None):
    p_posterior(np.ravel(samples), dist_label=label)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def enable_latex():
    plt.rcParams.update({"font.family": "serif", "font.serif": [], "font.sans-serif": ["DejaVuSans"]})


def trace(samples, plots=None, title=None, xlabel=None, ylabel=None, filename=None):
    ax = plt.gca()
    if plots is None:
        plot_trace(samples, title, xlabel, ylabel, ax)
    else:
        if title is not None:
            plt.suptitle(title)
        k = 1
        if len(plots) > 48:
            n_cols = 5
        elif len(plots) > 27:
            n_cols = 4
        elif len(plots) > 20:
            n_cols = 3
        elif len(plots) > 5:
            n_cols = 2
        else:
            n_cols = 1
        n_rows = int(math.ceil(len(plots) / n_cols))
        for i in range(len(plots)):
            if len(samples.shape) == 4:
                samples_i = samples[:, :, i, 0]
            elif len(samples.shape) == 3:
                samples_i = samples[:, :, i]
            else:
                print(samples)
                raise Exception("Invalid shape: " + samples.shape)
            sub = plt.subplot(n_rows, n_cols, k)
            plot_trace(samples_i, xlabel=xlabel, ylabel=plots[i], ax=sub)
            k += 1
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_trace(samples, title=None, xlabel=None, ylabel=None, ax=None):
    if len(samples.shape) == 3 and samples.shape[2] == 1:
        samples = samples[:, :, 0]
    ax.plot(samples)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)


if __name__ == "__main__":
    a = np.random.beta(np.array([[7, 13, 20], [10, 12, 8], [9, 14, 19]]),
                       np.array([[19, 24, 9], [5, 15, 10], [15, 16, 23]]),
                       size=(100, 3, 3))
    b = np.random.beta(np.array([7, 13, 20]), np.array([5, 15, 10]), size=(100, 3))
    matchup_matrix_posterior(a, b)
