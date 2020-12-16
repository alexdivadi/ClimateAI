from fastapi import FastAPI, File, UploadFile
from model import CNN_Model

app = FastAPI()
classes = {0: 'Cell', 1: 'Cell-Multi', 2: 'Cracking', 3: 'Diode', 4: 'Diode-Multi', 5: 'Hot-Spot', 6: 'Hot-Spot-Multi', 7: 'No-Anomaly', 8: 'Offline-Module', 9: 'Shadowing', 10: 'Soiling', 11: 'Vegetation'}
model = CNN_Model()

@app.get("/ping")
def pong():
    return {"ping":"pong!"}


@app.route("/predict", methods=["POST"])
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    try:
        data = img.imread(file)/255
        data = data.reshape(1, 40, 24, 1)
    except:
        return "Image is not correct size"
    prediction, _ = CNN_Model.predict(data)
    return classes.get(prediction)


@app.get("/")
async def root():
    return {"message": "Hello World"}
