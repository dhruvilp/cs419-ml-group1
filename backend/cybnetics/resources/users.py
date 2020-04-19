import jwt
import datetime
from flask import current_app

from .db import users_coll

class AlreadyExists(Exception):
    def __str__(self):
        return 'username already exists'

class BadUsernameOrPassword(Exception):
    def __str__(self):
        return 'wrong username or password'

def create(username, password):

    users = users_coll()
    is_user = users.find_one({'username': username})

    if is_user is None:
        hash_password = current_app.bcrypt.generate_password_hash(password)
        users.insert({'username': username, 'password': hash_password.decode('utf-8')})
        exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        token = jwt.encode({'username': username, 'exp': exp}, current_app.config['SECRET'])
        return token.decode('utf-8')
    else:
        raise AlreadyExists()


def check_password(username, password):
    users = users_coll()
    is_user = users.find_one({'username': username})
    if is_user:
        if current_app.bcrypt.check_password_hash(is_user['password'].encode('utf-8'), password):
            exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            token = jwt.encode({'username': username, 'exp': exp}, current_app.config['SECRET'])
            return token.decode('utf-8')
    raise BadUsernameOrPassword()


def check_jwt(token):
    try:
        return jwt.decode(token, current_app.config['SECRET'])
    except jwt.exceptions.InvalidTokenError:
        return None
