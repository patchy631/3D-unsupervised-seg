import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import laspy as lp
import pandas as pd
from sklearn.mixture import GaussianMixture


def read_df():
    df = pd.read_csv('../data/records.csv', usecols=['X', 'Y', 'Z', 'intensity', 'raw_classification'])
    pcd = o3d.io.read_point_cloud('')
    arr = df.to_numpy()[::10]
    labels = arr[:, -1]
    points = arr[:, :3] / 1000
    intensities = arr[:, 3]
    points = points[labels == 2]
    intensities = intensities[labels == 2]
    pcd.points.extend(points)
    inten_labels = GaussianMixture(n_components=3, random_state=0).fit_predict(np.expand_dims(intensities, axis=-1))
    return pcd, inten_labels, labels







def fit_planes_iteratively(pcd):
    # # RANSAC loop for multiple planar shapes detection
    # ******************************************************************************************************************
    segment_models = {}
    segments = {}
    max_plane_idx = 10
    rest = pcd
    for i in range(max_plane_idx):
        colors = plt.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(distance_threshold=0.000001, ransac_n=3, num_iterations=1000)
        # if len(inliers) < 1000:
        #     continue
        segments[i] = rest.select_by_index(inliers)
        segments[i].paint_uniform_color(list(colors[:3]))
        # o3d.visualization.draw_geometries([segments[i]])
        rest = rest.select_by_index(inliers, invert=True)
        print("pass", i, "/", max_plane_idx, "done.")
    print('RANSAC_loop_visualization')
    o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest])
    # *******************************************************************************************************************

if __name__ == '__main__':
    pcd, inten_labels, labels = read_df()
    fit_planes_iteratively(pcd)
