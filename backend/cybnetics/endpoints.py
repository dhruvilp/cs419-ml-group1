import json
import ast
import jwt
import datetime

from flask import Flask, jsonify, request, session

app = Flask(__name__)
app.config.from_pyfile('./config.py')


db = app.config['DATABASE']

# This is just a sanity check from myself
@app.route("/")
def get_initial_response():
    """Welcome message for the API."""
    # Message to the user
    message = {
        'apiVersion': 'v1.0',
        'status': '200',
        'message': 'Welcome to the Flask API'
    }
    # Making the message looks good
    resp = jsonify(message)
    # Returning the object
    return resp


@app.route("/login", methods=['POST'])
def login():
    """
    Login the user
    """

    users = db.users
    auth
    login_user = users.find_one({'username': request.json['username']})

    try:

        if login_user:
            if(login_user['password'].encode('utf-8') == request.json['password'].encode('utf-8')):
                exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
                token = jwt.encode({'username': request.json['username'], 'exp': exp}, app.config['SECRET_KEY'])
                return jsonify({'token': token.decode('utf-8'), 'valid_until': exp}), 200

        else:
            message = {'apiVersion': 'v1.0',
                        'status': '403',
                        'message': 'A user with those credentials does not exist!'}
            return jsonify(message), 403

    except Exception as e:
        print(e)
        message = {'apiVersion': 'v1.0',
                    'status': '400',
                    'message': 'Invalid request!'}
        return jsonify(message), 400



@app.route("/signup", methods=['POST', 'GET'])
def create_user():
    """
    Function to create new users.
    """

    collection = db.users

    if request.method == 'POST':
        existing_user = collection.find_one({'username': request.json['username']})


        if existing_user is None:

            try:
                # Create new users
                try:
                    body = ast.literal_eval(json.dumps(request.get_json()))
                except:
                    # Bad request as request body is not available
                    message = {'apiVersion': 'v1.0',
                                'status': '400',
                                'message': 'Invalid request!'}
                    return jsonify(message), 400

                record_created = collection.insert(body)

                # Prepare the response
                if isinstance(record_created, list):
                    # Return list of Id of the newly created item
                    return jsonify([str(v) for v in record_created]), 201
                else:
                    # Return Id of the newly created item
                    return jsonify(str(record_created)), 201
            except:
                # Error while trying to create the user
                message = {'apiVersion': 'v1.0',
                            'status': '500',
                            'message': 'Error while trying to create the user'}
                return jsonify(message), 500
        else:
            message = {'apiVersion': 'v1.0',
                        'status': '403',
                        'message': 'A user with that name already exists!'}
            return jsonify(message), 403



@app.route("/scoreboard", methods=['GET'])
def get_scores():
    """
    Function to get the scores for a user
    ** Still in progress
    """
    auth_token = request.json['token']
    decoded_message = jwt.decode(auth_token, app.config['SECRET_KEY'])
