from flask import Flask, render_template, request, Response
import os
import cv2
import warnings
from werkzeug.utils import secure_filename
from IPython import display
import ultralytics
import json

display.clear_output()
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'static/uploads'
WEIGHTS = "yolov8_e100\yolov8_e100_b32.pt"

prices = {'bottle':1200, 'keychain':150, 'lunchbox':3000, 'pencilcase':300, 'register':550, 'schoolbag':6000}
cart = {'bottle':0, 'keychain':0, 'lunchbox':0, 'pencilcase':0, 'register':0, 'schoolbag':0}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
camera=cv2.VideoCapture(0)
liveFeed = False
vidResult = []
model = YOLO(f'{WEIGHTS}')

def gen_frames():
    while True:
        global camera
        success, frame = camera.read()
        global liveFeed
        liveFeed = True

        if success:
            results = model(frame, save=True)
            
            annotatedFrame = results[0].plot()

            vidResult.append(results[0])

            ret, buffer = cv2.imencode('.jpeg', annotatedFrame)
            frame = buffer.tobytes()
        else:
            break

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/", methods=["GET", "POST"])
def welcome():
    return render_template("welcome.html")

@app.route("/live")
def live():
    global liveFeed, camera
    if not liveFeed:
        camera = cv2.VideoCapture(0)
    return render_template("live.html")

@app.route('/video')
def video():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/index')
def index():
    global liveFeed
    if liveFeed:
        global camera
        camera.release()
        liveFeed = False
    return render_template("index.html")

@app.route('/bill', methods=["GET", "POST"])
def bill():
    global liveFeed

    totalAmount = 0

    if liveFeed:
        global camera, cart
        camera.release()
        totalItems = []
        for result in vidResult:
            items = json.loads(result.tojson())

            for item in items:
                cart[item['name']] += 1

                if item['name'] not in totalItems:
                    totalItems.append(item['name'])

                totalAmount += prices[item['name']]

        liveFeed = False

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return render_template('bill.html', error='No file part')
        file = request.files['image']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('bill.html', error='No selected file')
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            result = model(filename, save=True)[0]

            # result.save_crop("results")

            items = json.loads(result.tojson())

            totalItems = []

            for item in items:
                cart[item['name']] += 1
                
                if item['name'] not in totalItems:
                    totalItems.append(item['name'])

                totalAmount += prices[item['name']]

    return render_template("bill.html", items=totalItems, prices=prices, cart=cart, totalAmount=totalAmount)


if __name__ == "__main__":
    app.run(debug=True)
