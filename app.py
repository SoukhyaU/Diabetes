from flask import Flask,render_template,request,session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
app = Flask(__name__)



@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/uploaddata',methods=["POST","GET"])
def uploaddata():
    if request.method=="POST":
        dataset = request.files['file']
        filename = dataset.filename
        file = "dataset\\"+filename
        session['dataset'] = file

        return render_template('upload.html', msg="success")
        return render_template('upload.html')

@app.route('/view')
def viewdata():
    datafile = session.get('dataset')
    df=pd.read_csv(datafile)
    df=df.head(10)
    return render_template('view.html',data=df.to_html())

@app.route('/split')
def splitdata():
    return render_template('split.html')

@app.route('/splitdata',methods=["POST","GET"])
def splitdataset():
    global x_train, y_train, x_test, y_test
    if request.method == "POST":
        testsize=request.form['testsize']
        testsize = float(testsize)
        datafile = session.get('dataset')
        df = pd.read_csv(datafile)
        x=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=testsize)
        len1 = len(x_train)
        len2 = len(x_test)

        return render_template('split.html',msg="done",tr=len1,te=len2)
    return render_template('split.html')
@app.route('/trainmodels')
def models():
    return render_template('trainmodels.html')

@app.route('/modelpredict',methods=["POST","GET"])
def prediction():
    if request.method == "POST":
        value = int(request.form['model'])

    if value==1:
        model = LogisticRegression()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        acc1=accuracy_score(y_test,y_pred)
        return render_template('trainmodels.html',msg="accuracy",acc=acc1,alg="LogisticRegression")
    if value == 2:
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc2 = accuracy_score(y_test,y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc2, alg="GaussianNB")
    if value == 3:
         model = SVC()
         model.fit(x_train, y_train)
         y_pred = model.predict(x_test)
         acc3 = accuracy_score(y_test, y_pred)
         return render_template('trainmodels.html', msg="accuracy", acc=acc3, alg="SVC")
    if value == 4:
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc4 = accuracy_score(y_test, y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc4, alg="DecisionTreeClassifier")
    if value == 5:
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc5 = accuracy_score(y_test, y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc5, alg="KNeighborsClassifier")
    if value == 6:
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc6 = accuracy_score(y_test, y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc6, alg="RandomForestClassifier")
    if value == 7:
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc7 = accuracy_score(y_test, y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc7, alg="GradientBoostingClassifier")
    if value == 8:
        model = XGBClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc8 = accuracy_score(y_test, y_pred)
        return render_template('trainmodels.html', msg="accuracy", acc=acc8, alg="XGBClassifier")

@app.route('/prediction')
def predict():
    return render_template('predict.html')

@app.route('/prediction1',methods =["POST","GET"])
def pred():
    s = []
    if request.method== "POST":
        pregnancies = request.form['pr']
        glucose = request.form['gl']
        BP = request.form['bp']
        skinthickness = request.form['skin']
        insulin = request.form['insulin']
        BMI = request.form['bmi']
        DPF = request.form['dpf']
        age = request.form['age']
        s.extend([pregnancies,glucose,BP,skinthickness,insulin,BMI,DPF,age])
        model = SVC()
        model.fit(x_train, y_train)
        y_pred = model.predict([s])
        return render_template('predict.html',msg="success",op=y_pred)


if __name__ == '__main__':
    app.secret_key="hai"
    app.run(debug=True)
