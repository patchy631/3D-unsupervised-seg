import numpy as np


def group_into_grids(points):
    """
    expects a numpy array of XYZ
    and divides it into a 10*10 grid fairly quickly
    and each grid can be accessed by it's index(0-99)
    its makes data analysis easy on a smaller samples
    :param points:
    :return: list of grids/tiles
    """
    pt_cloud_grids = []
    sorted_by_x = points[np.argsort(points[:, 0])]
    splits_along_x = np.array_split(sorted_by_x, 10)
    for split in splits_along_x:
        sorted_by_y = split[np.argsort(split[:, 1])]
        splits_along_y = np.array_split(sorted_by_y, 10)
        pt_cloud_grids.extend(splits_along_y)
    return pt_cloud_grids
