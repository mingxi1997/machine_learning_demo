
from flask import Flask,request,flash,redirect,url_for,render_template
# from flask_login import LoginManager, login_user, login_required, logout_user
 
from werkzeug.security import generate_password_hash





app = Flask(__name__,static_folder='',static_url_path='')

app.secret_key = "super secret key"



USERS = [
    {
        "id": 1,
        "name": 'lily',
        "password": generate_password_hash('123')
    },
    {
        "id": 2,
        "name": 'tom',
        "password": generate_password_hash('123')
    }
]


from werkzeug.security import check_password_hash


@app.route('/index', methods=['GET'])
def index():

   
    return render_template('index.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash('Invalid input.')
            return redirect(url_for('login'))

        # user = User.query.first()
        for user in USERS:
        # 验证用户名和密码是否一致
            if username == user.get('name') and check_password_hash(user.get('password'),password):
                # login_user(user)  # 登入用户
                # flash('Login success.')
                return redirect(url_for('index'))  # 重定向到主页

        flash('Invalid username or password.')  # 如果验证失败，显示错误消息
        return redirect(url_for('login'))  # 重定向回登录页面

    return render_template('login.html')

if __name__ == '__main__':
    app.run()
