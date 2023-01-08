import app_needs
from app_needs import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from flask import Flask, request, render_template, redirect, url_for
from flask_ngrok import run_with_ngrok
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Get the uploaded file
        files1 = request.files.getlist("file1")
        processed_file = process_file(files1)

        return redirect(url_for('success'))
          
    else:
        # Render the upload form
        return render_template("user_form.html")


def process_file(file):
    df=Load_Data(file)
    df=Days_last_rest(df)
    df=handi(df)
    df=dis_class(df)
    df=dis_finsh(df)
    global Horse_info
    All_hist,Horse_info=get_info(df)
    tests=All_hist[['Distance.1', 'Finish',
       'Venue.1', 'TrackCondition.1', 'Weight.1','Horse',
       'Jockey', 'HcpRating', 'rank',
       'Trainer', 'Age', 'Barrier', 'lastrestday',
       'handicaped', 'dis_class', 'Sectional_range',
       'Sectional_median', 'Previous_Sectional', 'MPS_range', 'MPS_median',
       'Previous_MPS']]
    tests.columns=['Distance', 'Finish',
       'Venue', 'TrackCondition', 'Weight','Horse',
       'Jockey', 'HcpRating', 'rank',
       'Trainer', 'Age', 'Barrier', 'lastrestday',
       'handicaped', 'dis_class', 'Sectional_range',
       'Sectional_median', 'Previous_Sectional', 'MPS_range', 'MPS_median',
       'Previous_MPS']
    cat_columns = tests.select_dtypes(['object']).columns
    global le
    le = LabelEncoder()
    tests[cat_columns] = tests[cat_columns].apply(lambda col: le.fit_transform(col))
    tests=tests.dropna()
    train=tests.drop(['Finish','rank'],axis=1)
    target=tests[['rank']]
    smote = SMOTE()

    # Oversample the minority class
    X_resampled, y_resampled = smote.fit_resample(train.values, target.values)
    global model
    global clf
    global model_1                                       
    
    best_params = {'depth': 8,
               'iterations': 800,
               'l2_leaf_reg': 30,
               'learning_rate': 0.01}
    model = CatBoostClassifier(
        **best_params,
        loss_function='MultiClass',        
    #         task_type="GPU",
        nan_mode='Min',
        verbose=False
    )

    model.fit(
        X_resampled, y_resampled,
        verbose_eval=200, 
        early_stopping_rounds=100,
        use_best_model=True,
        plot=True
    )

    clf = RandomForestClassifier(n_estimators=500, random_state=0,max_depth=8)

    # Train the classifier using the training data
    clf = clf.fit(X_resampled, y_resampled)

    model_1 = LogisticRegression(C=1.0, solver='lbfgs', penalty='l2')
    model_1 = model_1.fit(X_resampled, y_resampled)
    
    listed=[i['Horse'] for i in Horse_info]
    
    return listed

def predict(file):
  vb=pd.read_excel(file[0])
  try:
    vb=vb.drop(['Age'],axis=1)
  except:
    pass
  Horse=pd.DataFrame(Horse_info)
  test_2=pd.merge(vb,Horse,on='Horse',how='inner')
  test_2=Days_last_rest(test_2)
  test_2=handi(test_2)
  test_2=dis_class(test_2)
  test_2=test_2.dropna()
  test_1=test_2[['Distance', 
       'Venue', 'TrackCondition', 'Weight','Horse',
       'Jockey', 'HcpRating', 
       'Trainer', 'Age', 'Barrier', 'lastrestday',
       'handicaped', 'dis_class', 'Sectional_range',
       'Sectional_median', 'Previous_Sectional', 'MPS_range', 'MPS_median',
       'Previous_MPS']]
  cat_columns=['Venue', 'TrackCondition', 'Horse', 'Jockey', 'Trainer',
       'dis_class']
  test_1[cat_columns] = test_1[cat_columns].apply(lambda col: le.fit_transform(col))

  ans1=model_1.predict_proba(test_1)
  ans2=model.predict_proba(test_1)
  ans3=clf.predict_proba(test_1)
  test_2['top1']=[i[0]for i in ans1]
  test_2['top2']=[i[0]for i in ans2]
  test_2['top3']=[i[0]for i in ans3]
  test_2['top']=0.8*test_2['top1']+0.1*test_2['top1']+0.1*test_2['top1']
  ccc=test_2.sort_values('top',ascending=False)
  hj=[]
  for h in range(1,10):
    pl=ccc.loc[ccc.RaceNo==h][['Horse','Distance',  'Weight',
       'Trainer', 'Age', 'lastrestday',       
       'Sectional_median', 'Previous_Sectional','MPS_median',
       'Previous_MPS']]
    for z in pl[:3].values:
        hj+=[[h]+[i for i in z]]
  return hj

@app.route("/success", methods=["GET", "POST"])
def success():
  if request.method == "POST":
        # Get the uploaded file
        files2 = request.files.getlist("file2")
        processed_file = predict(files2)
        return render_template('final.html', list=processed_file)
  else:
      # Render the upload form
      return render_template("test_form.html") 
if __name__ == "__main__":
    app.run()
