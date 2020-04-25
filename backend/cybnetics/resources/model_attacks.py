import os

from bson import ObjectId

from .db import models_coll
from cybnetics import utils

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

def set_place(model_id, place, place_name):
    models_c = models_coll()
    models_c.update_one({
        '_id': model_id, # the attack must be a member of the target model
        'attacks.' + str(place): {'$exists': True}, # must have a attack at index place
        # the attack must not have its place assigned yet
        'attacks.' + str(place) + '.place': {'$exists': False}
    }, {
        '$set': {'attacks.'+str(place)+'.place': place_name} # assign the attack its place
    })

def save_attack(model_id, label, user, success):
    models_c = models_coll()
    attack_id = ObjectId()
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
    # see if inserted document needs to be assigned gold, silver, ect
    place = len(result['attacks'])
    if place <= 3:
        set_place(model_id, place, PLACE_MAP[place])
    attack = result['attacks'][place - 1]
    return attack
