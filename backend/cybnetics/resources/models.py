from bson import ObjectId
from bson.errors import InvalidId

from . import users
from .db import models_coll

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
    models = models_coll()
    aggregation = [{'$unwind': '$attacks'}]
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
        result[key] = result[key][0]['result']
    return result

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
    return models.remove(_id)

def is_owner(_id, user):
    model = find_one(_id)
    if not model:
        return False
    return model['owner'] == user
