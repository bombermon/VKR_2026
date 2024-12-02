from flask import Flask, render_template, redirect, url_for, request, session, flash
from flask_sqlalchemy import SQLAlchemy
import os




app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dating.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимум 16 MB



UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




# Импорт моделей
from models import User, Match

# Функция создания базы данных
def create_database():
    if not os.path.exists('dating.db'):
        with app.app_context():
            db.create_all()
        print("База данных создана!")

# Создаем базу данных при запуске приложения
create_database()

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Регистрация
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        bio = request.form['bio']
        file = request.files['photo']

        photo_path = None
        ml_predictions = None

        # Проверяем и сохраняем файл
        if file and allowed_file(file.filename):
            filename = f"id{len(User.query.all()) + 1}.{file.filename.rsplit('.', 1)[1].lower()}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            photo_path = os.path.join('uploads', filename)

            # Вызов ML-модели для анализа фотографии
            try:
                full_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                ml_predictions = analyze_photo_with_model(full_photo_path)
            except Exception as e:
                print(f"Ошибка при анализе изображения: {e}")

        # Сохраняем пользователя
        new_user = User(
            username=username,
            password=password,
            bio=bio,
            photo=photo_path,
            ml=str(ml_predictions) if ml_predictions else None
        )
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')



# Вход
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()

        if user:
            session['user_id'] = user.id
            flash('Вы вошли в систему!', 'success')
            return redirect(url_for('swipe'))

        flash('Неправильное имя пользователя или пароль.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    # Очистка данных сессии пользователя
    session.clear()
    return redirect(url_for('index'))
# Профиль
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        user.bio = request.form['bio']

        # Проверяем и сохраняем новую фотографию, если она загружена
        file = request.files['photo']
        if file and allowed_file(file.filename):
            filename = f"id{user.id}.{file.filename.rsplit('.', 1)[1].lower()}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user.photo = os.path.join('uploads', filename)

            # Вызов ML-модели для анализа новой фотографии
            try:
                full_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                user.ml = str(analyze_photo_with_model(full_photo_path))
            except Exception as e:
                print(f"Ошибка при анализе изображения: {e}")

        db.session.commit()
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)



# Свайпы
@app.route('/swipe', methods=['GET', 'POST'])
def swipe():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    current_user_id = session['user_id']

    # Обработка свайпов
    if request.method == 'POST':
        target_user_id = int(request.form['match_id'])
        liked = request.form['liked'] == 'true'

        # Сохраняем результат свайпа
        new_swipe = Match(user_id=current_user_id, target_user_id=target_user_id, liked=liked)
        db.session.add(new_swipe)
        db.session.commit()

    # Получаем список ID, которые текущий пользователь уже оценил
    evaluated_ids = db.session.query(Match.target_user_id).filter_by(user_id=current_user_id).all()
    evaluated_ids = [item[0] for item in evaluated_ids]

    # Находим пользователей, которых еще не оценил текущий пользователь
    potential_matches = User.query.filter(User.id.notin_(evaluated_ids), User.id != current_user_id).all()

    return render_template('swipe.html', potential_matches=potential_matches)
# Взаимные симпатии
@app.route('/matches')
def matches():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    current_user_id = session['user_id']

    # Находим всех пользователей, которые поставили лайк текущему пользователю
    # Мы ищем записи в таблице Swipe, где текущий пользователь был целевой (target_user_id)
    likes_received = db.session.query(Match).filter_by(target_user_id=current_user_id, liked=1).all()

    matched_users = []

    # Проходим по этим записям, чтобы проверить, поставил ли лайк текущий пользователь
    for like in likes_received:
        # Ищем, поставил ли текущий пользователь лайк этому пользователю
        mutual_like = db.session.query(Match).filter_by(user_id=current_user_id, target_user_id=like.user_id,
                                                        liked=1).first()

        if mutual_like:
            # Если да, это мэтч, добавляем пользователя в список
            user = User.query.get(like.user_id)
            matched_users.append(user)


    return render_template('matches.html', matches=matched_users)


if __name__ == '__main__':
    app.run(debug=True)
    create_database()