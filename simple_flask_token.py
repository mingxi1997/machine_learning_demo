from flask import Flask
from flask.globals import request
from flask.json import jsonify
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

app=Flask(__name__)
#私钥
app.config['SECRET_KEY']='LIUBINGZHEISBEST'
#忽略的检查的url
NOT_CHECK_URL=['/login']
#生产token
def create_token(api_user):
    '''
    生成token
    :param api_user:用户id
    :return: token
    '''
    
    #第一个参数是内部的私钥，这里写在共用的配置信息里了，如果只是测试可以写死
    #第二个参数是有效期(秒)
    s = Serializer(app.config["SECRET_KEY"],expires_in=100)
    #接收用户id转换与编码
    token = s.dumps({"id":api_user}).decode("ascii")
    return token
'''
装饰器模式
def login_required(view_func):
    @functools.wraps(view_func)
    def verify_token(*args,**kwargs):
        try:
            #在请求头上拿到token
            token = request.headers["z-token"]
        except Exception:
            #没接收的到token,给前端抛出错误
            #这里的code推荐写一个文件统一管理。这里为了看着直观就先写死了。
            return jsonify(msg = 'token为空')
        #解密
        s = Serializer(app.config["SECRET_KEY"])
        try:
            s.loads(token)
        except Exception:
            return jsonify(msg = "token已经过期")
        return view_func(*args,**kwargs)

    return verify_token
'''
@app.before_request
def login_required():
    if request.path not in NOT_CHECK_URL:
        try:
            token = request.headers["token"]
        except:
            return jsonify(msg='give me token')
        s = Serializer(app.config["SECRET_KEY"])
        try:
            s.loads(token)
        except:
            return jsonify(msg = "can not compute token")
@app.route('/login',methods=["POST"])
def info():
    '''
    POST携带userid来换取token
    代码省略其他逻辑
    '''
    return jsonify({'token':create_token(api_user=request.args.getlist('user_id'))})

@app.route('/msg')
def index():
    '''
    携带token登陆
    '''
    return jsonify(msg='allow you come')
if __name__ == "__main__":
    app.run()
