from flask import Flask, render_template, redirect, url_for, request
from model import CNN_Model

app = Flask(__name__)
model = CNN_Model()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/result/<prediction>")
def result(prediction):
    return "Predicted: %s" % prediction
    
@app.route("/error/<msg>")
def error(msg):
    return "Error: %s" % msg

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files['filename']
        extension = f.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return redirect(url_for('error', msg="File must be image!"))
        pred = model.predict(f)
        return redirect(url_for('result',prediction = pred))
    return render_template("upload.html")

if __name__ == '__main__':
    app.run(debug=True)
