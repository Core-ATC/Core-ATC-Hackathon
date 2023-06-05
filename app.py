from flask import Flask,render_template,url_for,request, jsonify, json
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
<<<<<<< HEAD:driver.py
	# return render_template('index.html')
	return ('Hello World')
@app.route('/predict',methods=['POST'])
=======
    return jsonify(
        json.dumps({
            "code": "SUCCESS",
            "message": "This route is working fine"
        })
    )
	# return render_template('home.html')

def ndarray_to_list(ndarray):
    # Convert ndarray to list
    if ndarray.ndim == 1:
        return ndarray.tolist()
    else:
        return [ndarray_to_list(arr) for arr in ndarray]

@app.route('/predict',methods=['GET', 'POST'])
>>>>>>> 5fc52cc20dda647285ecf44f52fe295da2796e3e:app.py
def predict():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		serialized_prediction = ndarray_to_list(my_prediction)
	# return render_template('result.html',prediction = my_prediction)
	return jsonify(
        json.dumps({
            "code": "SUCCESS",
            "prediction": json.dumps(serialized_prediction)
        })
    )

if __name__ == '__main__':
	app.run(debug=False)
