import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def diagnostic_plots(variable):
    """
    function takes an array as input
    Used in this project for analysis on intensity
    :param variable:
    :return: None
    """

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(variable, bins=100)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(variable, dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=variable)
    plt.title('Boxplot')

    plt.show()
