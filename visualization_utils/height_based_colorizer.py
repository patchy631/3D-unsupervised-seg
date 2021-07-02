import numpy as np
import pandas as pd
import pptk
import matplotlib.pyplot as plt


def read_df():
    df = pd.read_csv('../data/records.csv', usecols=['X', 'Y', 'Z', 'intensity', 'raw_classification'])
    arr = df.to_numpy()
    points = arr[:, :3] / 1000
    intensities = arr[:, 3]
    labels = arr[:, 4]
    return points, intensities, labels


def fit_multiple_horizontal_planes(num_planes):
    """
    colorize the point cloud accross multiple horizontal planes
    thru it
    :param num_planes: number of planes or horizontal sections
    :return:
    """
    points_o, intensities_o, labels_o = read_df()
    heights = points_o[:, -1]
    intervals = np.linspace(int(np.min(heights)), int(np.max(heights)), num_planes)
    color_labels = np.zeros((len(points_o),))
    for i in range(len(intervals) - 1):
        mask = np.bitwise_and((intervals[i] < heights), (heights < intervals[i + 1]))
        args_filtered = np.argwhere(mask)
        color_labels[args_filtered] = i
    max_label = np.max(color_labels)
    colors = plt.get_cmap("tab20")(color_labels / (max_label if max_label > 0 else 1))[:, :-1]
    v = pptk.viewer(points_o)
    v.attributes(colors)


if __name__ == '__main__':
    fit_multiple_horizontal_planes(15)
