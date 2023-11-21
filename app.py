from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("loan_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template('index_loan.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        gender = request.form['gender']
        marital_status = request.form['marital_status']
        no_of_dependents = request.form['no_of_dependents']
        education = request.form['education']
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])
        loan_amount_log = float(request.form['loan_amount_log'])
        total_income = float(request.form['total_income'])

        # Transform categorical variables
        gender = 1 if gender == 'Male' else 0
        marital_status = 1 if marital_status == 'Yes' else 0
        education = 1 if education == 'Graduate' else 0

        if no_of_dependents == '3+':
            no_of_dependents = 3
        else:
            no_of_dependents = int(no_of_dependents)

        # Make predictions
        prediction = model.predict(np.array([[gender, marital_status, no_of_dependents, education, loan_amount_term, credit_history, loan_amount_log, total_income]]))
        status = "Approved" if prediction[0] == 1 else "Not Approved"
        return render_template('index_loan.html', prediction_text=f'Loan Status: {status}')

if __name__ == '__main__':
    app.run(debug=True)
