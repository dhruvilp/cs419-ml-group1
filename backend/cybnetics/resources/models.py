from .db import models_coll
from bson import ObjectId
from bson.errors import InvalidId
from . import users

ATTACK_MODES = ['white', 'gray', 'black']
class BadAttackMode(Exception):
    def __init__(self, attack_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_mode = attack_mode
    def __str__(self):
        return 'invalid attack mode ' + self.attack_mode + ' must be in ' \
            + str(ATTACK_MODES)


def create(name, description, attack_mode, owner):
    if not attack_mode in ATTACK_MODES:
        raise BadAttackMode(attack_mode)
    models = models_coll()
    model = {
        '_id': ObjectId(),
        'name': name,
        'description': description,
        'attack_mode': attack_mode,
        'owner': owner,
        'ready': False
    }
    models.insert_one(model)
    return model

def scoreboard(username=None):
    pass

def find(query=None, attack_mode=None, user=None):
    models = models_coll()
    db_query = {}

    if attack_mode:
        if not attack_mode in ATTACK_MODES:
            raise BadAttackMode(attack_mode)
        db_query['attack_mode'] = attack_mode

    if user:
        if not users.exists(user):
            raise users.NoSuchUser(user)
        db_query['owner'] = user

    if query:
        db_query['$text'] = {'$search': query}

    return list(models.find(db_query))

def find_one(_id):
    models = models_coll()
    model = models.find_one({'_id': _id})
    return model

def remove(_id):
    models = models_coll()
    return models.remove_one(_id)

def is_owner(_id, user):
    model = find_one(_id)
    if not model:
        return False
    return model['owner'] == user
