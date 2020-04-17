import jwt
from flask import Flask, jsonify, request
from pymongo import MongoClient


app = Flask(__name__)
app.config.from_pyfile('./config.py')


@app.before_first_request
def db_connect():
    app.db = MongoClient(app.config['DB_URI']).get_default_database()


@app.route("/signup", methods=['POST'])
def create_user():
    """
    Function to create new users.
    **  Notice how I have to import create in the function since db_connect() is not called
        right away meaning that we have an app RuntimeError: Working outside of application context.
        in resources/db.py. It may not be a bad thing that we have to import here.
    """
    from cybnetics.resources.users import create

    try:

        data = request.get_json()
        username = data['username']
        password = data['password']
        response = create(username, password)
        return jsonify(response[0]), response[1]


    except:

        message = {'status': '400','message': 'Invalid input data.'}
        return jsonify(message), 400


@app.route("/login", methods=['POST'])
def login():
    """
    Login the user
    """
    from cybnetics.resources.users import check_password

    try:

        data = request.get_json()
        username = data['username']
        password = data['password']
        response = check_password(username, password)
        return jsonify(response[0]), response[1]

    except:

        message = {'status': '400','message': 'Invalid input data.'}
        return jsonify(message), 400
