"""Notebook-friendly plots of training_history; no training state is changed."""
import math


def plot_training_history(history, *, parameters=None, x_axis='tokens', title=None):
    """Return (figure, axes) for recorded optimizer updates.

    parameters: parameter names to compare, or None for all matrix parameters.
    x_axis: 'tokens' (millions of scored tokens) or 'updates'. Missing norm
    samples are omitted. Ratios for zero pre-step weights are omitted because
    their epsilon-based values are not useful relative-change measurements.
    """
    import matplotlib.pyplot as plt

    rows = list(history)
    if not rows:
        raise ValueError('history is empty; run training first')
    if x_axis not in ('tokens', 'updates'):
        raise ValueError("x_axis must be 'tokens' or 'updates'")
    key, scale, xlabel = (('scored_tokens', 1e6, 'Scored tokens (millions)')
                          if x_axis == 'tokens' else ('update', 1, 'Optimizer update'))
    x = lambda row: row[key] / scale
    samples = [row for row in rows if row.get('update_norms') is not None]
    available = dict.fromkeys(name for row in samples
                              for name in row['update_norms']['parameters'])
    if parameters is None:
        parameters = [name for name in available if any(
            row['update_norms']['parameters'].get(name, {}).get('parameter_kind') == 'matrix'
            for row in samples)]
    else:
        parameters = list(parameters)
        unknown = set(parameters) - available.keys()
        if unknown:
            raise ValueError(f'No norm samples for parameters: {sorted(unknown)}')

    legend_rows = math.ceil((len(parameters) + 1) / 3)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9 + .2 * legend_rows), constrained_layout=True)
    loss_ax, grad_ax, lr_ax, ratio_ax, norm_ax, speed_ax = axes.flat

    def line(ax, entries, field, label, *, log=False, **kwargs):
        points = [(x(row), values.get(field)) for row, values in entries]
        points = [(a, b) for a, b in points
                  if b is not None and math.isfinite(b) and (not log or b > 0)]
        if points:
            a, b = zip(*points)
            ax.plot(a, b, label=label, **kwargs)
            if log:
                ax.set_yscale('log')

    all_rows = [(row, row) for row in rows]
    line(loss_ax, all_rows, 'loss', 'Training update loss')
    line(grad_ax, all_rows, 'grad_norm', 'Gradient norm', log=True)
    line(grad_ax, all_rows, 'clip_threshold', 'Adaptive clipping threshold',
         log=True, color='tab:red', linestyle='--')
    clipped_rows = [(row, row) for row in rows if row.get('was_clipped')]
    line(grad_ax, clipped_rows, 'grad_norm', 'Clipped updates', log=True,
         color='tab:red', marker='x', linestyle='none')
    for index in range(max(len(row.get('learning_rates', [])) for row in rows)):
        line(lr_ax, [(row, {'lr': row['learning_rates'][index]}) for row in rows
                     if index < len(row.get('learning_rates', []))], 'lr', f'Group {index}')

    sampled = [(row, row['update_norms']) for row in samples]
    line(ratio_ax, [(row, value) for row, value in sampled if value['weight_norm'] > 1e-12],
         'update_to_weight_ratio', 'Global', log=True, color='black', linewidth=2)
    for name in parameters:
        entries = [(row, row['update_norms']['parameters'][name]) for row in samples
                   if name in row['update_norms']['parameters']]
        line(ratio_ax, [(row, value) for row, value in entries if value['weight_norm'] > 1e-12],
             'update_to_weight_ratio', name, log=True, marker='.', markersize=4)
    update_ax = norm_ax.twinx()
    line(norm_ax, sampled, 'weight_norm', 'Global weight norm', color='tab:blue')
    line(update_ax, sampled, 'update_norm', 'Global update norm', color='tab:orange')
    update_ax.set_ylabel('Update L2 norm (right)', color='tab:orange')
    update_ax.tick_params(axis='y', colors='tab:orange')
    norm_ax.tick_params(axis='y', colors='tab:blue')
    norm_ax.yaxis.label.set_color('tab:blue')
    # Linear axes show warmup from zero and independently resolve both trends.
    update_ax.set_ylim(bottom=0)

    # Snapshot copies affect sampled-step timing; show them as a separate series.
    for sampled_step, label in ((False, 'Unsampled steps'), (True, 'Norm-sampled steps')):
        line(speed_ax, [(row, row) for row in rows
                        if (row.get('update_norms') is not None) == sampled_step],
             'tokens_per_second', label, marker='.', linestyle='none', alpha=.6)

    for ax, heading, ylabel in zip(axes.flat,
            ('Training loss', 'Gradient norm (before clipping)', 'Learning rate',
             'Actual update / pre-step weight norm', 'Global weight and update norms',
             'Training throughput'),
            ('Cross entropy', 'L2 norm', 'Learning rate', 'Ratio', 'Weight L2 norm (left)', 'Scored tokens / second')):
        ax.set(title=heading, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=.25)
        if ax is ratio_ax and ax.lines:
            fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
                       ncols=3, fontsize=7, title='Update-to-weight ratios: all selected matrices')
        elif ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No recorded data', ha='center', va='center', transform=ax.transAxes)
    norm_lines = norm_ax.lines + update_ax.lines
    if norm_lines:
        norm_ax.legend(norm_lines, [item.get_label() for item in norm_lines], fontsize=7)
    fig.suptitle(title or 'Training diagnostics')
    return fig, axes
