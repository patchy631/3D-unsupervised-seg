import laspy as lp
import pandas as pd


def write_las_file(file_path, points):
    """
    Writes numpy array as a las file
    :param file_path:
    :param points:
    :return: None
    """
    header = lp.header.LasHeader()
    las_data = lp.lasdata.LasData(header)
    las_data.x = points[:,0]
    las_data.y = points[:,1]
    las_data.Z = points[:,2]
    las_data.write(file_path)

def make_df(point_cloud):
    """
    Reads a las file and extracts all important parameters
    such as XYZ, intensity, classification etc. and create
    a data frame out of it
    :param point_cloud:
    :return: None
    """
    arr = point_cloud.points.array
    columns = list(dict(arr.dtype.fields).keys())
    l = list((map(lambda x: list(x), arr[::10])))
    df = pd.DataFrame(l, columns=columns)
    df.to_csv('../data/records.csv', index=False)

if __name__ == '__main__':
     point_cloud = lp.read('C_37EZ1.las')
    make_df(point_cloud)
