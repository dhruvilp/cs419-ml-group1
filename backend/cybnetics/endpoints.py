from flask import Flask, jsonify, request
from flask_bcrypt import Bcrypt
from pymongo import MongoClient

from .resources import db, users
from .utils import *


app = Flask(__name__)
app.config.from_pyfile('./config.py')


@app.before_first_request
def setup():
    app.db = MongoClient(app.config['DB_URI']).get_default_database()
    app.bcrypt = Bcrypt(app)

@app.route('/signup', methods=['POST'])
@require_json_body
def create_user():
    """
    Function to create new users.
    **  Notice how I have to import create in the function since db_connect() is not called
        right away meaning that we have an app RuntimeError: Working outside of application context.
        in resources/db.py. It may not be a bad thing that we have to import here.
    """

    try:
        data = request.get_json()
        username = data['username']
        password = data['password']
    except Exception as e:
        return 'missing parameter' + str(e), 400

    try:
        token = users.create(username, password)
        return jsonify({'token': token}), 201
    except users.AlreadyExists as e:
        return str(e), 401

@app.route('/login', methods=['POST'])
@require_json_body
def login():
    """
    Login the user
    """

    try:
        data = request.get_json()
        username = data['username']
        password = data['password']
    except Exception as e:
        return 'missing parameter' + str(e), 400

    try:
        token = users.check_password(username, password)
        return jsonify({'token': token}), 201
    except users.BadUsernameOrPassword as e:
        return str(e), 401
