import numpy as np
import pandas as pd
import pptk
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from GMM_impl import gaussian_mixture_model

def read_df():
    df = pd.read_csv('data/records.csv', usecols=['X', 'Y', 'Z', 'intensity', 'raw_classification'])
    arr = df.to_numpy()
    points = arr[:, :3] / 1000
    intensities = arr[:, 3]
    labels = arr[:, 4]
    return points, intensities, labels


def segment_trees_and_buildings():
    """
    Segments non ground points by doing a segregation based on intensity
    using Gaussing mixture model.
    Assumption: We mainly have trees and buildings as non ground objects
    and their LiDAR intensities come from 2 different distributions.
    :return: None
    """
    points_o, intensities_o, labels_o = read_df()
    heights = points_o[:,2]
    points = points_o[heights > 8]
    intensities = intensities_o[heights > 8]
    labels = GaussianMixture(n_components=2, random_state=0).fit_predict(np.expand_dims(intensities, axis=-1))
    max_label = max(labels)
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))[:, :-1]
    v = pptk.viewer(points)
    v.attributes(colors)


if __name__ == '__main__':
    segment_trees_and_buildings()
