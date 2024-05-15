import pandas as pd
import numpy as np
import re
import string
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load

# Load Fake and True news data
data_fake = pd.read_csv("C:/Users/verma/Downloads/DBMS/Fake.csv")
data_true = pd.read_csv("C:/Users/verma/Downloads/DBMS/True.csv")

# Add class labels to distinguish fake and true news
data_fake["class"] = 0
data_true["class"] = 1

data_fake.shape, data_true.shape

data_fake_manual_testing = data_fake.tail(10).copy()  # Make a copy to avoid modifying the original DataFrame
for i in range(23470, 23460, -1):
    data_fake.drop([i], axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10).copy()  # Make a copy to avoid modifying the original DataFrame
for i in range(21406, 21396, -1):
    data_true.drop([i], axis=0, inplace=True)

data_fake.shape, data_true.shape

data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

# Concatenate fake and true news data
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['subject'], axis=1)
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index', 'title'], axis=1, inplace=True)


# Prepare data for training
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorize the text data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train classifiers or load from saved models
try:
    LR = load("LR_model.joblib")
    DT = load("DT_model.joblib")
    GB = load("GB_model.joblib")
    RF = load("RF_model.joblib")
except FileNotFoundError:  # If models are not found, train and save them
    LR = LogisticRegression()
    DT = DecisionTreeClassifier()
    GB = GradientBoostingClassifier(random_state=0)
    RF = RandomForestClassifier()

    LR.fit(xv_train, y_train)
    DT.fit(xv_train, y_train)
    GB.fit(xv_train, y_train)
    RF.fit(xv_train, y_train)

    # Save the trained models
    dump(LR, "LR_model.joblib")
    dump(DT, "DT_model.joblib")
    dump(GB, "GB_model.joblib")
    dump(RF, "RF_model.joblib")


# Function to preprocess text and make predictions
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return pred_LR[0], pred_DT[0], pred_GB[0], pred_RF[0]


# Flask app setup
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result_LR = None
    result_DT = None
    result_GB = None
    result_RF = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        pred_LR, pred_DT, pred_GB, pred_RF = manual_testing(user_input)
        result_LR = "Fake News" if pred_LR == 0 else "True News"
        result_DT = "Fake News" if pred_DT == 0 else "True News"
        result_GB = "Fake News" if pred_GB == 0 else "True News"
        result_RF = "Fake News" if pred_RF == 0 else "True News"

    return render_template("index.html", result_LR=result_LR, result_DT=result_DT, result_GB=result_GB,
                           result_RF=result_RF)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
