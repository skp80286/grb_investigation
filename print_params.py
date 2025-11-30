params = {'jetType': 'tophat', 'z': 2.011, 'logn0': 0, 'p': 2.07, 's': 0, 'loglf': 3.0, 'A': 0.0}

# Format params dictionary in ASCII table format
def format_dict_table(d, title="Parameters"):
    """Format a dictionary in a nice ASCII table format and return as string"""
    if not d:
        return "Empty dictionary"

    # Format values: floats use .4f, others as strings
    formatted_values = {}
    for key, value in d.items():
        if isinstance(value, float):
            formatted_values[key] = f"{value:.4f}"
        else:
            formatted_values[key] = str(value)

    # Find the maximum width for keys and formatted values
    max_key_width = max(len(str(k)) for k in d.keys())
    max_val_width = max(len(v) for v in formatted_values.values())

    # Set minimum column widths
    key_width = max(max_key_width, len("Parameter"))
    val_width = max(max_val_width, len("Value"))

    # Create the table
    border = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"
    header = f"| {'Parameter':<{key_width}} | {'Value':<{val_width}} |"

    # Build table string
    lines = []
    lines.append(f"\n{title}")
    lines.append(border)
    lines.append(header)
    lines.append(border)

    for key, value in d.items():
        formatted_val = formatted_values[key]
        lines.append(f"| {str(key):<{key_width}} | {formatted_val:<{val_width}} |")

    lines.append(border)
    lines.append("")

    return "\n".join(lines)

def format_parameters_table(median_params, rel_sigma_params, priors):
    """
    Format median parameters and priors as an ASCII table string.

    Parameters:
    -----------
    median_params : dict
        Dictionary with parameter names as keys and posterior median values as values
    rel_sigma_params : dict
        Dictionary with parameter names as keys and posterior relative sigma values as values
    priors : dict
        Dictionary with parameter names as keys and dict with 'low' and 'high' as values

    Returns:
    --------
    str : Formatted ASCII table as a string
    """
    import numpy as np

    # Get all unique parameter names
    all_params = set(median_params.keys())
    all_params.update(priors.keys())

    # Filter out non-numeric parameters (like 'jetType')
    numeric_params = []
    for param in sorted(all_params):
        if param in median_params:
            value = median_params[param]
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit()):
                numeric_params.append(param)

    if not numeric_params:
        return "No numeric parameters found"

    # Calculate column widths
    param_width = max(max(len(str(p)) for p in numeric_params), len("Parameter"))
    prior_low_width = max(len("Prior - low"), 12)
    prior_high_width = max(len("Prior - high"), 12)
    median_width = max(len("Posterior Median"), 15)
    rel_sigma_width = max(len("Posterior Sigma(rel))"), 15)

    # Create header
    header = f"| {'Parameter':<{param_width}} | {'Prior - low':<{prior_low_width}} | {'Prior - high':<{prior_high_width}} | {'Posterior Median':<{median_width}} | {'Posterior Sigma(rel)':<{rel_sigma_width}} |"
    border = "+" + "-" * (param_width + 2) + "+" + "-" * (prior_low_width + 2) + "+" + "-" * (prior_high_width + 2) + "+" + "-" * (median_width + 2) + "+" + "-" * (rel_sigma_width + 2) + "+"

    # Build table string
    lines = []
    lines.append(border)
    lines.append(header)
    lines.append(border)

    # Build rows
    for param in numeric_params:
        # Get prior bounds
        prior_low = "N/A"
        prior_high = "N/A"

        # Check if parameter exists in priors (handle log transformations)
        prior_key = param
        if param not in priors:
            # Try without 'log' prefix (e.g., logthc -> thc)
            if param.startswith('log'):
                prior_key = param[3:]  # Remove 'log' prefix

        if prior_key in priors:
            prior_info = priors[prior_key]
            if isinstance(prior_info, dict):
                prior_low = prior_info.get('low', 'N/A')
                prior_high = prior_info.get('high', 'N/A')
                # If prior is log_uniform and param is in log space, convert bounds to log space
                if prior_info.get('prior_type') == 'log_uniform' and param.startswith('log'):
                    if isinstance(prior_low, (int, float)) and isinstance(prior_high, (int, float)):
                        if prior_low > 0 and prior_high > 0:
                            prior_low = np.log10(prior_low)
                            prior_high = np.log10(prior_high)
                        else:
                            prior_low = "-inf" if prior_low == 0 else str(prior_low)
                            prior_high = "-inf" if prior_high == 0 else str(prior_high)
            elif isinstance(prior_info, (list, tuple)) and len(prior_info) >= 2:
                prior_low = prior_info[0]
                prior_high = prior_info[1]

            if prior_low == prior_high: continue
        else:
            continue

        # Format values
        if isinstance(prior_low, (int, float)):
            prior_low_str = f"{prior_low:.4f}"
        elif isinstance(prior_low, str) and prior_low == "-inf":
            prior_low_str = "-inf"
        else:
            prior_low_str = str(prior_low)

        if isinstance(prior_high, (int, float)):
            prior_high_str = f"{prior_high:.4f}"
        elif isinstance(prior_high, str) and prior_high == "-inf":
            prior_high_str = "-inf"
        else:
            prior_high_str = str(prior_high)

        # Get median value
        median_val = median_params.get(param, 'N/A')
        if isinstance(median_val, (int, float)):
            median_str = f"{median_val:.4f}"
        else:
            median_str = str(median_val)

        # Get sigma value
        sigma_val = rel_sigma_params.get(param, 'N/A')
        if isinstance(sigma_val, (int, float)):
            sigma_str = f"{sigma_val:.4f}"
        else:
            sigma_str = str(sigma_val)

        lines.append(f"| {str(param):<{param_width}} | {prior_low_str:<{prior_low_width}} | {prior_high_str:<{prior_high_width}} | {median_str:<{median_width}} | {sigma_str:<{rel_sigma_width}} |")

    lines.append(border)

    return "\n".join(lines)

if __name__ == "__main__":
    priors_uniform = {
        "loge0": {"low": 53.75, "high": 53.75, "prior_type": "uniform"},
        "logepsb": {"low": -5, "high": -1, "prior_type": "uniform"},
        "logepse": {"low": -1.5, "high": -0.5, "prior_type": "uniform"},
        "logn0": {"low": -2, "high": 0.0, "prior_type": "uniform"},
        "thc": {"low": 0.08, "high": 0.3, "prior_type": "log_uniform"},  # radians
        "thv": {"low": 0.0, "high": 0.1, "prior_type": "log_uniform"},
        "p": {"low": 2.2, "high": 2.2, "prior_type": "uniform"},
        "s": {"low": 4, "high": 4, "prior_type": "uniform"},
        "loglf": {"low": 5, "high": 5, "prior_type": "uniform"},
        "A": {"low": 0.0, "high": 0.0},  ## fix at 0
    }
    median_params = {'jetType': 'tophat', 'z': 2.011, 'logn0': 0, 'p': 2.07, 's': 0, 'loglf': 3.0, 'A': 0.0, 'loge0': 53.70432471174358, 'logepsb': -2.1332383740487253, 'logepse': -1.144602182702135, 'logthc': -1.0245131572950257, 'logthv': -1.3909562597529446}
    sigma_params = {'logn0': 0.1, 'p': 0.07, 's': 0, 'loglf': 0.0, 'loge0': 0.070432471174358, 'logepsb': 0.1332383740487253, 'logepse': 0.0144602182702135, 'logthc': 0.00245131572950257, 'logthv': 0.3909562597529446}
    
    # Get the table as a string
    table_str = format_parameters_table(median_params, sigma_params, priors_uniform)
    print(table_str)
    
    table_str = format_dict_table(params, "Parameters Dictionary")
    print(table_str)

