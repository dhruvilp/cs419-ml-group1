import copy

from bson import ObjectId
from bson.errors import InvalidId

from . import users
from .db import models_coll
from cybnetics.model_builder import make_model_class

ATTACK_MODES = ['white', 'gray', 'black']
class BadAttackMode(Exception):
    def __init__(self, attack_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_mode = attack_mode
    def __str__(self):
        return 'invalid attack mode ' + self.attack_mode + ' must be in ' \
            + str(ATTACK_MODES)

MODEL_TYPES = ['mnist', 'cifar']
class BadModelType(Exception):
    def __init__(self, model_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
    def __str__(self):
        return 'invalid model type ' + self.model_type + ' must be in ' \
            + str(MODEL_TYPES)

def create(name=None, description=None, attack_mode=None, owner=None,
           layers=None, color=None, pools=[], dropouts=[], **kwargs):
    for param in [name, description, attack_mode, owner, layers, color]:
        if param is None:
            raise ValueError('missing required param to create')

    if not attack_mode in ATTACK_MODES:
        raise BadAttackMode(attack_mode)
    # validate by making class and see if it throws an error
    make_model_class(copy.deepcopy(layers), copy.deepcopy(pools), copy.deepcopy(dropouts))
    models = models_coll()
    model = {
        '_id': ObjectId(),
        'name': name,
        'description': description,
        'layers': layers,
        'pools': pools,
        'dropouts': dropouts,
        'color': bool(color),
        'attack_mode': attack_mode,
        'owner': owner,
        'ready': False
    }
    models.insert_one(model)
    return model

def scoreboard(username=None):
    models = models_coll()
    aggregation = [{'$match': {'ready': True}},
                   {'$unwind': '$attacks'}]
    if username:
        if not users.exists(username):
            raise users.NoSuchUser
        aggregation += [{'$match': {'attacks.user': username}}]

    def count(match):
        return [{'$match': match},
                {'$count': 'result'}]

    aggregation += [{'$facet': {
        'total_successes': count({'attacks.success': True}),
        'total_attempts': count({}),
        'gold_medals': count({'attacks.place': 'gold'}),
        'silver_medals': count({'attacks.place': 'silver'}),
        'bronze_medals': count({'attacks.place': 'bronze'}),
        'users': [
            {'$group': {
                '_id': {
                    'user': '$attacks.user',
                    'model': '$_id'
                },
                'name': {'$first': '$name'},
                'attempts': {'$sum': 1},
                'successes': {'$sum': {'$cond': {
                    'if': '$attacks.success',
                    'then': 1,
                    'else': 0
                }}},
                'points_earned': {'$sum': '$attacks.points'}
            }},
            {'$group': {
                '_id': '$_id.user',
                'attacked_models': {'$push': {
                    '_id': '$_id.model',
                    'name': '$name',
                    'attempts': '$attempts',
                    'sucesses': '$sucesses',
                    'points_earned': '$points_earned'
                }},
                'total_points': {'$sum': '$points_earned'},
                'total_attempts': {'$sum': '$attempts'},
                'total_successes': {'$sum': '$successes'}
            }}
        ]
    }}]
    result = list(models.aggregate(aggregation))
    # output massaging due to $facet
    result = result[0]
    for key in ['total_successes', 'total_attempts',
                'gold_medals', 'silver_medals', 'bronze_medals']:
        try:
            result[key] = result[key][0]['result']
        except:
            result[key] = 0
    return result

def find(query=None, attack_mode=None, user=None, ready=True):
    models = models_coll()
    db_query = {}

    if ready is not None:
        db_query['ready'] = ready

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

    return list(models.find(db_query, projection={
        'layers': False,
        'pools': False
    }))

def find_one(_id):
    models = models_coll()
    model = models.find_one({'_id': _id})
    return model

def remove(_id):
    models = models_coll()
    return models.remove(_id)

def is_owner(_id, user):
    model = find_one(_id)
    if not model:
        return False
    return model['owner'] == user

def set_ready(_id):
    models_coll().update_one({'_id': _id}, {'$set': {'ready': True}})
