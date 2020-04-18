import jwt
import datetime
from flask import current_app
from flask_bcrypt import Bcrypt
from .db import db


bcrypt = Bcrypt(current_app)


def create(username, password):

    users = db.users
    is_user = users.find_one({'username': username})

    if is_user is None:
        hash_password = bcrypt.generate_password_hash(password)
        users.insert({'username': username, 'password': hash_password.decode('utf-8')})
        exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        token = jwt.encode({'username': username, 'exp': exp}, current_app.config['SECRET'])
        return token.decode('utf-8')
    else:
        raise Exception('The username already exists.')


def check_password(username, password):
    users = db.users
    is_user = users.find_one({'username': username})
    if is_user:
        if bcrypt.check_password_hash(is_user['password'].encode('utf-8'), password):
            exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            token = jwt.encode({'username': username, 'exp': exp}, current_app.config['SECRET'])
            return token.decode('utf-8')
        else:
            raise Exception('Wrong password.')
    else:
        raise Exception('Username not found.')


def check_jwt(jwt):
    pass
