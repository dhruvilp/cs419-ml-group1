import os
ADMINS = ['roofus', 'doofus'] # change this to add yourself as an admin
SECRET = 'foewpkfweopfew' # secret used for pyjwt login tokens and sessions
DB_URI = 'mongodb://localhost/cybnetics'
# getcwd means the uploads folder should be in the cybnetics folder
UPLOADS = os.path.join(os.path.dirname(__file__), 'uploads')
