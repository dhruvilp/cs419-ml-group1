from flask import Flask, jsonify, request
from flask_bcrypt import Bcrypt
from pymongo import MongoClient, TEXT
from bson import ObjectId
from bson.errors import InvalidId

from .resources import db, users, models, model_images
from .utils import *


app = Flask(__name__)
app.config.from_pyfile('./config.py')


@app.before_first_request
def setup():
    app.db = MongoClient(app.config['DB_URI']).get_default_database()
    db.models_coll().create_index([('name', TEXT), ('description', TEXT)])
    app.json_encoder = MongoJSONEncoder
    app.bcrypt = Bcrypt(app)

@app.route('/')
def server_test():
    return jsonify("Server is running!")

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

@app.route('/models', methods=['POST'])
@require_json_body
@require_body_jwt
def create_model(user=None):
    """endpoint for creating models"""
    try:
        data = request.get_json()
        name = data['name']
        description = data['description']
        attack_mode = data['attack_mode']
    except Exception as e:
        return 'missing parameter' + str(e), 400

    try:
        model = models.create(name, description, attack_mode, user)
        return jsonify(model)
    except models.BadAttackMode as e:
        return str(e), 400

@app.route('/models', methods=['GET'])
@require_json_body
@require_body_jwt
def find_models(user=None):
    """endpoint for searching through models"""

    data = request.get_json()
    query = data.get('query')
    target_user = data.get('user')
    attack_mode = data.get('attack_mode')

    try:
        result = models.find(query=query, attack_mode=attack_mode, user=target_user)
    except users.NoSuchUser as e:
        return str(e), 404
    except models.BadAttackMode as e:
        return str(e), 400

    if len(result) == 0:
        return 'No models matched that query', 404
    return jsonify(result)

@app.route('/models/<_id>/model', methods=['POST'])
@require_url_jwt
def upload_model(_id, user=None):
    if not request.files.get('model'):
        return 'missing file named "model"', 400
    f = request.files['model']
    try:
        _id = ObjectId(_id)
        if not model_images.can_store(_id, user):
            return 'you don\'t own that model', 403
        model_images.store(_id, f)
    except InvalidId:
        return 'invalid model id', 400
    return '', 204

@app.route('/models/<_id>/model', methods=['GET'])
@require_url_jwt
def download_model(_id, user=None):
    try:
        _id = ObjectId(_id)
        if not model_images.can_get(_id, user): # todo what if id not exists
            return 'You lack permissions needed to download that model', 403

        r = model_images.get(_id)
        print(r)
        return r
    except InvalidId:
        return 'invalid model id', 400
    return '', 204
