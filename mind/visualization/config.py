"""
Centralized configuration for all visualizations ensuring consistency.

This module defines colors, display names, figure sizes, and styling functions
used consistently across all visualization components.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

# Define consistent color scheme
SIGNAL_COLORS = {
    'calcium_signal': '#356d9e',  #  blue
    'deltaf_signal': '#4c8b64',  #  green
    'deconv_signal': '#a85858'  #  red
}

# Display names for signals
SIGNAL_DISPLAY_NAMES = {
    'calcium_signal': 'Calcium',
    'deltaf_signal': 'ΔF/F',
    'deconv_signal': 'Deconvolved'
}

# Gradient colors for heatmaps
SIGNAL_GRADIENTS = {
    'calcium_signal': ['#f0f4f9', '#c6dcef', '#7fb0d3', '#356d9e'],
    'deltaf_signal': ['#f6f9f4', '#d6ead9', '#9dcaa7', '#4c8b64'],
    'deconv_signal': ['#fdf3f3', '#f0d0d0', '#d49c9c', '#a85858']
}

# Model display names
MODEL_DISPLAY_NAMES = {
    'random_forest': 'Random Forest',
    'svm': 'SVM',
    'mlp': 'MLP',
    'fcnn': 'FCNN',
    'cnn': 'CNN',
    'modified_cnn': 'Modified CNN'
}

# Standard figure sizes
FIGURE_SIZES = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (16, 12),
    'grid_5x3': (12, 20),
    'grid_1x3': (18, 6)
}


def set_publication_style():
    """
    This function configures matplotlib to produce plots with a consistent appearance.
    """
    plt.style.use('seaborn-v0_8-white')
    sns.set_style("white")
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

    logger.debug("Set publication-quality plot styling")


def get_signal_colormap(signal_type):
    """
    Get a custom colormap for a specific signal type.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if signal_type not in SIGNAL_GRADIENTS:
        logger.warning(f"Unknown signal type: {signal_type}, using default colormap")
        return plt.cm.viridis

    gradient = SIGNAL_GRADIENTS[signal_type]
    return LinearSegmentedColormap.from_list('custom', gradient)

