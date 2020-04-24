from flask_cors import CORS
from cybnetics import app

CORS(app)
app.run(port=5000, debug=True)
