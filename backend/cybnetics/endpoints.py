from flask import Flask, jsonify, request
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from pymongo import MongoClient, TEXT
from bson import ObjectId
from bson.errors import InvalidId

from .resources import db, users, models, \
    model_images, model_attacks, model_datasets
from .utils import *
from .model_builder import BadModelSpec


app = Flask(__name__)
CORS(app)
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
    Create the user
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
@require_admin
def create_model(user=None):
    """endpoint for creating models"""
    try:
        data = request.get_json()
        name = data['name']
        description = data['description']
        layers = data['layers']
        # pools is optional
        attack_mode = data['attack_mode']
    except Exception as e:
        return 'missing parameter' + str(e), 400

    try:
        model = models.create(owner=user, **data)
        return jsonify(model)
    except models.BadAttackMode as e:
        return str(e), 400
    except models.BadModelType as e:
        return str(e), 400
    except BadModelSpec as e:
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
    ready = True
    if target_user == user:
        ready = None

    try:
        result = models.find(query=query,
                             attack_mode=attack_mode,
                             user=target_user,
                             ready=ready)
    except users.NoSuchUser as e:
        return str(e), 404
    except models.BadAttackMode as e:
        return str(e), 400

    if len(result) == 0:
        return 'No models matched that query', 404
    return jsonify(result)

@app.route('/models/<_id>', methods=['GET'])
@require_json_body
@require_body_jwt
def get_model(_id, user=None):
    try:
        _id = ObjectId(_id)
        model = models.find_one(_id)
        if not model:
            return 'no model found', 404
        if not (model['attack_mode'] == 'white' or model['owner'] == user):
            del model['pools']
            del model['layers']
        return jsonify(model)
    except InvalidId:
        return 'invalid model id', 400

@app.route('/models/<_id>', methods=['DELETE'])
@require_json_body
@require_body_jwt
def remove_model(_id, user=None):
    try:
        _id = ObjectId(_id)
        if not models.is_owner(_id, user):
            return 'you do not own that model', 403
        models.remove(_id)
        model_images.remove(_id)
        model_datasets.remove(_id)
        return '', 204
    except InvalidId:
        return 'invalid model id', 400

@app.route('/models/<_id>/model', methods=['POST'])
@require_url_jwt
@require_admin
def upload_model(_id, user=None):
    f = request.files.get('model')
    if not f:
        return 'missing file named "model"', 400
    try:
        _id = ObjectId(_id)
        if not model_images.can_store(_id, user):
            return 'you don\'t own that model', 403
        model_images.store(_id, f)
    except InvalidId:
        return 'invalid model id', 400
    except model_images.BadModelFormat:
        return 'model state_dict was not in pytorch format', 415
    return '', 204

@app.route('/models/<_id>/model', methods=['GET'])
@require_json_body
@require_body_jwt
def download_model(_id, user=None):
    try:
        _id = ObjectId(_id)
        if not model_images.can_get(_id, user): # todo what if id not exists
            return 'You lack permissions needed to download that model', 403

        return model_images.get(_id)
    except InvalidId:
        return 'model not found', 404
    return '', 204

@app.route('/models/<_id>/dataset', methods=['POST'])
@require_url_jwt
@require_admin
def upload_dataset(_id, user=None):
    f = request.files.get('dataset')
    if not f:
        return 'missing file named "dataset"', 400
    try:
        _id = ObjectId(_id)
        if not model_datasets.can_store(_id, user):
            return 'you don\'t own that model', 403
        model_datasets.store(_id, f)
    except InvalidId:
        return 'invalid model id', 400
    return '', 204

@app.route('/models/<_id>/dataset', methods=['GET'])
@require_json_body
@require_body_jwt
def download_dataset(_id, user=None):
    try:
        _id = ObjectId(_id)
        if not model_datasets.can_get(_id, user): # todo what if id not exists
            return 'You lack permissions needed to download that model', 403

        return model_datasets.get(_id)
    except InvalidId:
        return 'model not found', 404
    return '', 204

@app.route('/models/<_id>/attack', methods=['POST'])
@require_url_jwt
def attempt_attack(_id, user=None):
    try:
        _id = ObjectId(_id)
        f = request.files.get('image')
        label = request.args.get('label')
        if not f:
            return 'missing file named "image"', 400
        if not label:
            return 'missing url parameter "label"', 400
        if not models.find_one(_id):
            return 'model not found', 404
        success = model_attacks.simulate_attack(_id, label, f, user)
        result = model_attacks.save_attack(_id, label, user, success)
        return jsonify(result)
    except InvalidId:
        return 'invalid model id', 400
    except model_attacks.InvalidAttack as e:
        return str(e), 400

@app.route('/scoreboard', methods=['GET'])
@require_json_body
@require_body_jwt
def scoreboard(user=None):
    data = request.get_json()
    search_user = data.get('username')
    try:
        scoreboard_data = models.scoreboard(username=search_user)
    except users.NoSuchUser as e:
        return str(e), 400
    return jsonify(scoreboard_data)
