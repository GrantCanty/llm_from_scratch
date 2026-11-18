import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from decoder.plotting import plot_training_history


class PlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_sparse_samples_and_zero_weight_ratios(self):
        rows = []
        for update in range(1, 5):
            value = dict(weight_norm=2., update_norm=.02,
                         update_to_weight_ratio=.01, parameter_kind='matrix')
            sample = dict(value, parameters={'matrix.weight': value, 'bias': dict(
                value, weight_norm=0., update_to_weight_ratio=1e10,
                parameter_kind='bias_or_norm')})
            rows.append(dict(update=update, scored_tokens=update * 1000,
                             loss=3 / update, grad_norm=.5, learning_rates=[.001, .002],
                             tokens_per_second=100, update_norms=sample if update % 2 else None))
        fig, axes = plot_training_history(rows, parameters=['matrix.weight', 'bias'])
        self.assertEqual(len(axes.flat), 6)
        ratio = axes[1, 0]
        self.assertEqual(len(ratio.lines), 2)  # global and matrix; zero bias omitted
        self.assertEqual(list(ratio.lines[0].get_xdata()), [.001, .003])
        self.assertEqual(len(axes[0, 2].lines), 2)
        weight_ax = axes[1, 1]
        update_ax = fig.axes[-1]
        self.assertEqual(weight_ax.get_yscale(), 'linear')
        self.assertEqual(update_ax.get_yscale(), 'linear')
        self.assertEqual(weight_ax.lines[0].get_label(), 'Global weight norm')
        self.assertEqual(update_ax.lines[0].get_label(), 'Global update norm')
        self.assertEqual(update_ax.get_ylabel(), 'Update L2 norm (right)')
        fig.canvas.draw()

    def test_legacy_history_and_validation(self):
        history = [dict(update=1, scored_tokens=1000, loss=2., grad_norm=1.,
                        learning_rates=[.001], tokens_per_second=100)]
        fig, axes = plot_training_history(history, x_axis='updates')
        self.assertEqual(len(axes[1, 0].lines), 0)
        fig.canvas.draw()
        for args, kwargs in (([], {}), (history, {'x_axis': 'bad'}),
                             (history, {'parameters': ['unknown']})):
            with self.assertRaises(ValueError):
                plot_training_history(args, **kwargs)
