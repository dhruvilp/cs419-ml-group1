from functools import wraps

from flask import request

from .resources import users

def require_body_jwt(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        data = request.get_json()
        if not data.get('token'):
            return 'Auth token missing from request', 400
        if not users.check_jwt(data['token']):
            return 'Invalid or expired auth token', 401
        return f(*args, **kwargs)
    return wrapper

def require_url_jwt(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.args.get('token'):
            return 'Auth token missing from request', 400
        if not users.check_jwt(request.args['token']):
            return 'Invalid or expired auth token', 401
        return f(*args, **kwargs)
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
