from flask import Flask, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64
from model import CNN_Model

app = Flask(__name__)
app.secret_key = "secret key"
model = CNN_Model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        f = request.files['filename']
        extension = f.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            flash("Extension not supported")
            return redirect(request.url)
        try:
            pred = model.predict_anomaly(f)
            flash("Predicted class: %s" % pred)
        except:
            flash("File must be 40x24 pixels")
            return redirect(request.url)
        filename = secure_filename(f.filename)
        im = Image.open(f)
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        return render_template("index.html", filename=filename, img_data=encoded_img_data.decode('utf-8'))
    return render_template("index.html")
'''
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
'''
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
        try:
            pred = model.predict_anomaly(f)
        except:
            return redirect(url_for('error', msg="File must be 40x24 pixels"))
        return redirect(url_for('result',prediction = pred))
    return render_template("upload.html")

if __name__ == '__main__':
    app.run(debug=True)
