from flask import Flask, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename
from model import CNN_Model

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
model = CNN_Model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['filename']
        extension = f.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            flash("File must be image!")
            return redirect(request.url)
        try:
            pred = model.predict_anomaly(f)
            flash("Predicted class: %s" % pred)
        except:
            flash("File must be 40x24 pixels")
            return redirect(request.url)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template("index.html", filename=filename)
    return render_template("index.html")

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

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
