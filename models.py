from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    bio = db.Column(db.Text, nullable=True)
    photo = db.Column(db.String(120), nullable=True)  # Поле для фотографии
    ml = db.Column(db.Text, nullable=True)  # Поле для хранения предсказаний модели



class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    target_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    liked = db.Column(db.Boolean, nullable=False)  # True - нравится, False - не нравится
