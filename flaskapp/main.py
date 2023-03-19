from flask import Flask, request, render_template
import requests
import json

app = Flask(__name__)
api_url = "http://localhost:5000/predict"

@app.route('/', methods = ['GET'])
def home():
    return render_template(r"index.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files["myfile"]
        image = file.read()
        response = requests.post(api_url, files={'file': image})
        result = json.loads(response.text)
        print(result)
    return render_template("prediction.html", confidence=result['Confidence'], prediction=result['Class'])
    
        
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)