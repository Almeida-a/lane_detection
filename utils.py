import os


def get_abs_path(rel_path):
    """

    :param rel_path:
    :return:
    """
    return os.path.realpath(rel_path)
