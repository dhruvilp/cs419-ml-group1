from os import path

from bson.errors import InvalidId
from flask import current_app, send_file

from .db import models_coll
from . import models
import cybnetics.utils as utils

class ModelNotUploaded(Exception):
    pass

class AccessDenied(Exception):
    pass

def store(_id, model_image_file):
    model = models.find_one(_id)
    if not model:
        return InvalidId()
    model_image_file.save(utils.get_path('m_' + str(_id)))
    if model['attack_mode'] == 'white' \
       and not utils.upload_exists('d_' + str(_id)):
        return
    models_coll().update_one({'_id': _id}, {'$set': {'ready': True}})


def get(_id):
    if not utils.upload_exists('m_' + str(_id)):
        raise ModelNotUploaded()
    path = utils.get_path('m_' + str(_id))
    print(path)
    return send_file(path)

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
