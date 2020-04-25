import os
from os import path

from bson.errors import InvalidId
from flask import current_app, send_file
import torch

from .db import models_coll
from . import models
import cybnetics.utils as utils


class ModelNotUploaded(Exception):
    pass

class AccessDenied(Exception):
    pass

class BadModelFormat(Exception):
    pass

def model_filename(_id):
    return utils.get_path('m_' + str(_id) + '.pt')

def store(_id, model_image_file):
    model = models.find_one(_id)
    if not model:
        return InvalidId()
    filename = model_filename(_id)
    model_image_file.save(filename)
    try:
        torch.load(filename)
    except:
        os.remove(filename)
        raise BadModelFormat

    if model['attack_mode'] == 'white' \
       and not utils.upload_exists('d_' + str(_id)):
        return
    models_coll().update_one({'_id': _id}, {'$set': {'ready': True}})


def get(_id):
    filename = model_filename(_id)
    if not path.exists(filename):
        raise ModelNotUploaded()
    return send_file(filename)

def remove(_id):
    try:
        os.remove(utils.get_path('m_' + str(_id)))
    except FileNotFoundError:
        pass

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
    owner = model['owner'] == user
    return owner or white_mode
