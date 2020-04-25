import os

from bson import ObjectId
from pymongo import ReturnDocument

from cybnetics import utils
from .db import models_coll
from . import models

SCORE_MAP = {
    'white': 1,
    'gray': 5,
    'black': 10,
}

PLACE_MAP = {
    1: 'gold',
    2: 'silver',
    3: 'bronze'
}

class InavlidImage(Exception):
    pass

def simulate_attack(model_id, label, attack_image, user):
    filename = utils.get_path('a_' + str(model_id) + '_' + str(ObjectId()))
    # store attack image
    attack_image.save(filename)
    # run the model with the image to see if its defeated
    success = True

    # if we cant load the image or the format is bad
    # throw InvalidImage()

    # delete the image
    os.remove(filename)
    return success

def set_place(attack_id, place_name):
    models_c = models_coll()
    models_c.update_one({
        'attacks._id': attack_id
    }, {
        '$set': {'attacks.$.place': place_name} # assign the attack its place
    })

def save_attack(model_id, label, user, success):
    models_c = models_coll()
    attack_id = ObjectId()
    points = 0
    if success:
        model = models.find_one(model_id)
        points = SCORE_MAP[model['attack_mode']]
    # add attack to model document in db.
    result = models_c.find_one_and_update({
        '_id': model_id
    }, {
        '$push': {
            'attacks': {
                '_id': attack_id,
                'label': label,
                'user': user,
                'success': success,
                'points': points
            }
        }
    }, return_document=ReturnDocument.AFTER)
    attack = result['attacks'][-1]
    # see if inserted document needs to be assigned gold, silver, ect
    # lmao this gets slower with more and more attacks
    place = len(list(filter(lambda attack: attack['success'], result['attacks'])))
    if place <= 3 and success:
        attack['place'] = PLACE_MAP[place]
        set_place(attack_id, PLACE_MAP[place])
    return attack
