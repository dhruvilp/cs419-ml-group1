import os

from .db import db
from cybnetics import utils

def dataset_filename(_id):
    return utils.get_path('d_' + str(_id) + '.zip')


def store(model_image_file):
    pass

def get(model_id, user):
    pass

def remove(_id):
    try:
        os.remove(dataset_filename(_id))
    except FileNotFoundError:
        pass
