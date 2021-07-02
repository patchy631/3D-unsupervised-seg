import matplotlib.pyplot as plt
import numpy as np
import pptk
import pandas as pd


def read_df():
    """
    reads data from a df which is
    created from raw las file
    ref: utility/las_utils.py
    :return:
    """
    df = pd.read_csv('../data/records.csv', usecols=['X', 'Y', 'Z', 'intensity', 'raw_classification'])
    arr = df.to_numpy()
    points = arr[:, :3]/1000
    intensities = arr[:, 3]
    labels = arr[:,4]
    return points, intensities, labels

def visualize_pointcloud(param=None):
    """
    Visualize the entire point cloud
    based on raw classification values
    :param param:
    :return:
    """
    points, intensities, labels = read_df()
    if param == 'intensity':
        intensities = np.clip(intensities, a_min=1, a_max=255)
        max_intensity = max(intensities)
        colors = plt.get_cmap("tab20")(intensities / (max_intensity if max_intensity > 0 else 1))[:, :-1]
    else:
        max_label = max(labels)
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))[:, :-1]
    v = pptk.viewer(points)
    v.attributes(colors)


if __name__ == '__main__':
    visualize_pointcloud()
