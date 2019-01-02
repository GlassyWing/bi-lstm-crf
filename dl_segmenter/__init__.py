from dl_segmenter.core import DLSegmenter
import json

get_or_create = DLSegmenter.get_or_create


def save_config(obj, config_path, encoding="utf-8"):
    with open(config_path, mode="w+", encoding=encoding) as file:
        json.dump(obj.get_config(), file)
