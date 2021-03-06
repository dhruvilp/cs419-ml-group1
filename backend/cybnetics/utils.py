from datetime import datetime, date
from functools import wraps
from os import path

from bson import ObjectId
from flask import request, current_app
from flask.json import JSONEncoder

from .resources import users

def require_login(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Cybnetics-Token')
        if not token:
            return 'missing Cybnetics-Token header in request', 400
        decoded = users.check_jwt(token)
        if not decoded:
            return 'Invalid or expired auth token', 401
        return f(*args, user=decoded['username'], **kwargs)
    return wrapper


def require_admin(f):
    @wraps(f)
    def wrapper(*args, user=None, **kwargs):
        if not user in current_app.config['ADMINS']:
            return 'admin required for this endpoint', 403
        return f(*args, user=user, **kwargs)
    return wrapper

def require_content_type(typ):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            print(request.content_type, typ)
            if request.content_type != typ:
                return 'expected Content-Type: ' + typ, 415
            return f(*args, **kwargs)
        return wrapper
    return decorator

require_json_body = require_content_type('application/json')

class MongoJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.iso_format()
        if isinstance(o, ObjectId):
            return str(o)
        else:
            return super().default(o)

def get_path(filename):
    return path.join(current_app.config['UPLOADS'], filename)

def upload_exists(filename):
    return path.exists(get_path(filename))

def dataset_filename(_id):
    return get_path('d_' + str(_id) + '.zip')

def model_filename(_id):
    return get_path('m_' + str(_id) + '.pt')
