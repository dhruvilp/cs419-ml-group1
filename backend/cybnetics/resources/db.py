from flask import current_app
# must be set before using other resources
def db():
    return current_app.db

def coll(name):
    return db()[name]

def users_coll():
    return coll('users')
