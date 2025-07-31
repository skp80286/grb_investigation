import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.interpolate import CubicSpline

def plot_lightcurve(title, csv_filename, x_col, y_col, error_col, frequency_col='Frequency Band', 
fit_spline=False, invert_y=False, selected_bands=None, yscale='linear', xscale='log'):
    """
    Create a professional publication-quality light curve plot from CSV data.
    
    Parameters:
    -----------
    title: str
        Title for the plot
    csv_filename : str
        Path to the CSV file containing the data
    x_col : str
        Name of the column to use for x-axis (time)
    y_col : str
        Name of the column to use for y-axis (magnitude)
    error_col : str
        Name of the column to use for error bars
    frequency_col : str, optional
        Name of the column containing frequency band information (default: 'Frequency Band')
    fit_spline : bool, optional
        Whether to fit and display cubic splines for each frequency band (default: False)
    """
    
    # Set up professional styling for research publication
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    rcParams['axes.linewidth'] = 1.2
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 11
    rcParams['figure.dpi'] = 300
    
    # Load and parse CSV data
    data = []
    headers = []
    
    try:
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)  # Get column headers
            
            # Find column indices
            try:
                x_idx = headers.index(x_col)
                y_idx = headers.index(y_col)
                error_idx = headers.index(error_col)
                freq_idx = headers.index(frequency_col)
            except ValueError as e:
                print(f"Error: Column not found in CSV headers: {e}")
                print(f"Available columns: {headers}")
                return
            
            # Read data rows
            for row in csv_reader:
                if len(row) >= max(x_idx, y_idx, error_idx, freq_idx) + 1:
                    data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Parse and clean data
    parsed_data = []
    for row in data:
        try:
            # Parse x value (time)
            x_val = row[x_idx].strip()
            if x_val and x_val != '':
                # Handle scientific notation
                if 'E' in x_val or 'e' in x_val:
                    x_val = float(x_val)
                else:
                    x_val = float(x_val)
            else:
                continue
            
            # Parse y value (magnitude)
            y_val = row[y_idx].strip()
            if y_val and y_val != '':
                y_val = float(y_val)
            else:
                continue
            
            # Parse error value
            error_val = row[error_idx].strip()
            if error_val and error_val != '':
                error_val = float(error_val)
            else:
                error_val = None
            
            # Parse frequency band
            freq_val = row[freq_idx].strip()
            if freq_val and freq_val != '':
                parsed_data.append({
                    'x': x_val,
                    'y': y_val,
                    'error': error_val,
                    'frequency': freq_val
                })
        except (ValueError, IndexError):
            continue
    
    if not parsed_data:
        print("Error: No valid data found in CSV file.")
        return
    
    # Group data by frequency band
    frequency_bands = {}
    for item in parsed_data:
        freq = item['frequency']
        if freq not in frequency_bands:
            frequency_bands[freq] = []
        frequency_bands[freq].append(item)
    
    # Sort data within each frequency band by x value
    for freq in frequency_bands:
        frequency_bands[freq].sort(key=lambda x: x['x'])
    
    # Define rainbow colors for frequency bands (red to violet)
    # Order: J (maroon) -> z (dark red) -> r/R (red) -> v/V (green) -> b (blue) -> u (violet)
    rainbow_colors = {
        'J': '#8B0000',      # Dark Red/Maroon
        'z': '#B22222',      # Fire Brick (deeper red)
        'r': '#FF0000',      # Red
        'R': '#FF0000',      # Red (alternative)
        'r\'': '#FF0000',    # Red (alternative)
        'v': '#00FF00',      # Green
        'V': '#00FF00',      # Green (alternative)
        'b': '#0000FF',      # Blue
        'u': '#8000FF',      # Violet
        'i': '#FF00FF',      # Magenta (for i-band)
        'i\'': '#FF00FF',    # Magenta (for i-band alternative)
        'g': '#00FFFF',      # Cyan (for g-band)
    }
    
    # Fallback colors for any other frequency bands
    fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create the figure with professional styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each frequency band with different colors
    for i, (freq, band_data) in enumerate(frequency_bands.items()):
        if selected_bands is not None and freq not in selected_bands: continue
        if len(band_data) > 0:
            # Extract x and y values
            x_vals = [item['x'] for item in band_data]
            y_vals = [item['y'] for item in band_data]
            
            # Get color for this band based on frequency
            if freq in rainbow_colors:
                color = rainbow_colors[freq]
            else:
                # Use fallback color for unknown frequency bands
                color = fallback_colors[i % len(fallback_colors)]
                print(f"Warning: Unknown frequency band '{freq}', using fallback color")
            
            # Plot the line
            ax.plot(x_vals, y_vals, marker='o', markersize=6, linewidth=2, 
                    color=color, label=freq, alpha=0.8)
            
            # Fit and plot cubic spline if requested
            if fit_spline and len(x_vals) >= 3:
                try:
                    # Sort data for spline fitting
                    sorted_indices = np.argsort(x_vals)
                    x_sorted = np.array([x_vals[i] for i in sorted_indices])
                    y_sorted = np.array([y_vals[i] for i in sorted_indices])
                    
                    # Create cubic spline
                    cs = CubicSpline(x_sorted, y_sorted)
                    
                    # Generate smooth curve points
                    x_smooth = np.logspace(np.log10(min(x_sorted)), np.log10(max(x_sorted)), 100)
                    y_smooth = cs(x_smooth)
                    
                    # Plot the spline fit as dashed line
                    ax.plot(x_smooth, y_smooth, '--', color=color, linewidth=1.5, alpha=0.7)
                    
                except Exception as e:
                    print(f"Warning: Could not fit spline for frequency band '{freq}': {e}")
            
            # Add error bars if available
            error_vals = []
            error_x_vals = []
            error_y_vals = []
            
            for item in band_data:
                if item['error'] is not None:
                    error_vals.append(item['error'])
                    error_x_vals.append(item['x'])
                    error_y_vals.append(item['y'])
            
            if error_vals:
                ax.errorbar(error_x_vals, error_y_vals, yerr=error_vals,
                           fmt='none', color=color, capsize=3, capthick=1, alpha=0.7)
    
    # Customize the plot for publication quality
    ax.set_xlabel(x_col, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=14, fontweight='bold')
    ax.set_title('GRB 250704B', fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis to log scale for better visualization of GRB data
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    # Invert y-axis (high values at bottom, low values at top)
    if invert_y: ax.invert_yaxis()
    
    # Customize ticks
    ax.tick_params(which='major', length=8, width=1.2, direction='in')
    ax.tick_params(which='minor', length=4, width=1.0, direction='in')
    ax.tick_params(axis='both', which='both', top=True, right=True)
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Customize grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
              fontsize=11, bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_filename = csv_filename.replace('.csv', '_lightcurve.pdf').replace('data/', 'plots/')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    
    # Show the plot
    plt.show()
    
    # Print some statistics
    print(f"Data loaded successfully!")
    print(f"Number of data points: {len(parsed_data)}")
    print(f"Frequency bands found: {list(frequency_bands.keys())}")
    
    x_vals_all = [item['x'] for item in parsed_data]
    y_vals_all = [item['y'] for item in parsed_data]
    
    print(f"Time range: {min(x_vals_all):.1e} to {max(x_vals_all):.1e} seconds")
    print(f"Magnitude range: {min(y_vals_all):.2f} to {max(y_vals_all):.2f}")
    print(f"Plot saved as: {output_filename}")

# Example usage
if __name__ == "__main__":
    # Example call for the GRB data
    plot_lightcurve(
        title='GRB 250704B',
        csv_filename='data/GRB 250704B - lightcurve.csv',
        x_col='t-t0 (seconds)',
        y_col='Magnitude (AB)',
        error_col='Error bar (Mag)',
        frequency_col='Frequency Band',
        fit_spline=True  # Set to True to show cubic spline fits
    ) 