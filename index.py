from flask import Flask, render_template, request
import joblib
import numpy as np

dtc=joblib.load(open('models/DecisionTreeClassifier.sav', 'rb'))
dtr=joblib.load(open('models/DecisionTreeRegressor.sav', 'rb'))

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def f():
    answer=0
    ans=0
    gen= request.form.get('GENDER',type=int)
    hei= request.form.get('HEIGHT', type=float)
    wei= request.form.get('WEIGHT', type=float)
    model = request.form.get('MODELS')
    if hei==None:
        hei =0
    else:
        hei=hei * 30.38
    if wei==None:
        wei=0
    else:
        wei =wei
    a = [[gen], [hei],[wei]]
    a = np.array(a)
    b = np.transpose(a)
    if model == 'Decission Tree classifier':
        ans=dtc.predict(b)[0]
    elif model == 'Decission Tree REGRESSOR':
        ans=dtr.predict(b)[0]
    hei=hei / 30.38
    answer=round(ans,0)
    return render_template('index.html', answer=answer,HEIGHT=hei,WEIGHT=wei)


if __name__ == '__main__':
    app.run(debug=True)
