import os
from os import path

from bson.errors import InvalidId
from flask import current_app, send_file

from . import models
from cybnetics.utils import dataset_filename, model_filename

class DatasetNotUploaded(Exception):
    pass

class AccessDenied(Exception):
    pass

def store(_id, model_image_file):
    model = models.find_one(_id)
    if not model:
        return InvalidId()
    filename = dataset_filename(_id)
    model_image_file.save(filename)

    if path.exists(model_filename(_id)):
        models.set_ready(_id)

def get(_id):
    filename = dataset_filename(_id)
    if not path.exists(filename):
        raise DatasetNotUploaded()
    return send_file(filename)


def can_store(_id, user):
    model = models.find_one(_id)
    if not model:
        return False
    return model['owner'] == user

def can_get(_id, user):
    model = models.find_one(_id)
    if not model:
        return False
    white_mode = model['attack_mode'] == 'white'
    gray_mode = model['attack_mode'] == 'gray'
    owner = model['owner'] == user
    return owner or white_mode or gray_mode


def remove(_id):
    try:
        os.remove(dataset_filename(_id))
    except FileNotFoundError:
        pass
