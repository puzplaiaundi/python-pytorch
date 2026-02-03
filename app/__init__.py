from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    from app.api.ia import ia_bp
    app.register_blueprint(ia_bp)

    return app
