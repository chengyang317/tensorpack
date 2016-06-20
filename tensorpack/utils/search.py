# -*- coding: UTF-8 -*-
# File: search.py
# Author: philipcheng
# Time: 6/19/16 -> 5:22 PM
import re

__all__ = ['get_elem_from_dict']


def get_elem_from_dict(var_dict, key):
    """
    Get dict elem by key or a regex key
    :param var_dict:
    :param key: maybe a full name or a regex
    :return: matched elem
    """
    elem = var_dict.get(key, None)
    if elem is None:
        keys = var_dict.keys()
        matched_keys = []
        for item in keys:
            if re.search(key, item):
                matched_keys.append(key)
        assert len(matched_keys) <= 1, "{} matched multi keys {}".format(key, ','.join(matched_keys))
        if len(matched_keys) == 0:
            return None
        elem = var_dict[matched_keys[0]]
    return elem
