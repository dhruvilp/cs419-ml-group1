from flask import Flask, jsonify

app = Flask(__name__)
app.config.from_pyfile('./config.py')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'hello': 'world'})
