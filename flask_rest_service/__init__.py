import os
from flask.ext import restful
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from flask import make_response
from bson.json_util import dumps
from werkzeug.utils import secure_filename

app = Flask(__name__)

def output_json(obj, code, headers=None):
    resp = make_response(dumps(obj), code)
    resp.headers.extend(headers or {})
    return resp

DEFAULT_REPRESENTATIONS = {'application/json': output_json}

api = restful.Api(app)
api.representations = DEFAULT_REPRESENTATIONS

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            f_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(f_path)
            picture = sk_io.imread(f_path)
            faceAndEmotion = detectEmotionsFromPicture(picture)
            return redirect('/')

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


from skimage import io as sk_io
import mxnet.ndarray as nd
from skimage.feature import hog
from mxnet.gluon import nn
from skimage.transform import resize
import numpy as np
import face_recognition

num_oris = 4

net = nn.Sequential()
net.add(nn.Conv2D(channels=24, kernel_size=4, in_channels=4, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=120, kernel_size=4, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(300, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(2))

net.load_parameters('net.params')

def imageToHog(img_, num_oris):
    img = resize(img_, (144,144))
    fd, hog_img = hog(img, orientations = num_oris, cells_per_block = (1,1),
                      visualise = True, feature_vector = False, multichannel = True)
    np_fd = np.array(fd)
    squeezed = np.squeeze(np_fd, axis = [2,3]).reshape((num_oris,18,18))
    return squeezed

def detectEmotionFromFace(face_img):
    hog = imageToHog(face_img,num_oris)
    hog_np = nd.array(np.array([hog]))
    res = net(hog_np).argmax(axis=1)
    return res.asnumpy()[0]

def detectEmotionsFromPicture(image):
    locs = face_recognition.face_locations(image)
    if(len(locs)>0):
        [a,b,c,d] = locs[0]
        face = image[a:c,d:b]
        happy = detectEmotionFromFace(face)
        return (True, happy)
    return (False, -1)


