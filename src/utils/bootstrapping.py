import numpy as np

def calculate_bootstrap_statistics(metrics_dict, confidence_level=0.95):
    """
    Calculate statistical summaries (mean, standard deviation, and confidence intervals)
    for multiple metrics across bootstrap iterations.

    Args:
        metrics_dict (dict):
            A dictionary mapping an identifier (e.g., "bootstrap_0", "bootstrap_1", ...)
            to another dictionary of metrics. For example:
                {
                    'bootstrap_0': {'mae': 0.1, 'mse': 0.01, 'rmse': 0.1, ...},
                    'bootstrap_1': {'mae': 0.12, 'mse': 0.012, 'rmse': 0.11, ...},
                    ...
                }
        confidence_level (float, optional):
            The confidence level to use for the interval. Defaults to 0.95 (95% CI).

    Returns:
        dict:
            A dictionary that contains, for each metric, a dictionary of summary statistics:
            {
                'mae': {
                    'mean': (float),
                    'std': (float),
                    'lower_ci': (float),
                    'upper_ci': (float)
                },
                'mse': {
                    'mean': (float),
                    'std': (float),
                    'lower_ci': (float),
                    'upper_ci': (float)
                },
                ...
            }
    """
    # Edge case: if metrics_dict is empty, return an empty dictionary
    if not metrics_dict:
        return {}

    # Extract the list of metric names from the first bootstrap entry
    first_key = next(iter(metrics_dict.keys()))
    metric_names = list(metrics_dict[first_key].keys())

    # Initialize a dictionary to hold lists of metric values across all bootstrap iterations
    aggregated_metrics = {m: [] for m in metric_names}

    # Collect metric values across all bootstrap iterations
    for _, metrics in metrics_dict.items():
        for m in metric_names:
            aggregated_metrics[m].append(metrics[m])

    # Set up confidence interval boundaries
    alpha = 1 - confidence_level
    lower_bound = alpha / 2
    upper_bound = 1 - alpha / 2

    # Calculate summary statistics for each metric
    final_metrics = {}
    for m in metric_names:
        data = np.array(aggregated_metrics[m])
        mean_val = np.mean(data)
        std_val = np.std(data)
        ci_lower = np.quantile(data, lower_bound)
        ci_upper = np.quantile(data, upper_bound)

        final_metrics[m] = {
            "mean": mean_val,
            "std": std_val,
            "lower_ci": ci_lower,
            "upper_ci": ci_upper
        }

    return final_metrics
