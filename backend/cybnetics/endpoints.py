from flask import Flask, jsonify
from pymongo import MongoClient

app = Flask(__name__)
app.config.from_pyfile('./config.py')

@app.before_first_request
def db_connect():
    app.db = MongoClient(app.config['DB_URI']).get_default_database()

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'hello': 'world'})
