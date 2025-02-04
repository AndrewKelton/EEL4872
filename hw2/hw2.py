# Andrew Kelton
# February 15, 2025
# EEL4872 Spring 2025
# Professor Gurupur
# Assignment 2

# Correlational Heatmap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # Read in data
    data = pd.read_csv('data.csv')
    data.set_index('Variables', inplace=True)

    corr_mat = data.corr() # Get correlations

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Relationship Between Cancer and Symptoms of Patients")

    plt.show() # Show heatmap