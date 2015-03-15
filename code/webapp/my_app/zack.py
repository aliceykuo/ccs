from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

# OUR HOME PAGE
#============================================
@app.route('/')
def welcome():
    myname = "Zack"
    return render_template('beach_result.html', data=myname)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)