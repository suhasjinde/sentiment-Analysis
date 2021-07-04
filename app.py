import re
from flask import Flask, render_template,request,url_for,redirect,flash
from google.protobuf import message
import keras
import numpy as np
import librosa

app = Flask(__name__)
app.secret_key="AvdscisWQ3cso_!!!!!3221@dcsncaoWcncjs122412vjjsa"

class livePredictions:
    def __init__(self, path, file):
        self.path = path
        self.file = file
    def load_model(self):
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()
    def makepredictions(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        return self.convertclasstoemotion(predictions)

    @staticmethod
    def convertclasstoemotion(pred):
        label_conversion = {'0': 'You seem: Neutral, try being more expressive!',
                            '1': 'You seem: Calm, just like the sea',
                            '2': 'You seem: Happy! Im glad.',
                            '3': 'You seem: Sad, Just know that things are going to be okay',
                            '4': 'You seem: Angry, you should calm down and smile more',
                            '5': 'You seem: Fearful',
                            '6': 'You seem: Disgust',
                            '7': 'You seem: Surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['wav']

@app.route('/processAudio',methods=['GET','POST'])
def processAudio():
    if request.method=='POST':
        audioFIle = request.files['audiofile']
        if audioFIle and allowed_file(audioFIle.filename):
            pred = livePredictions(path='SER_model.h5',file=audioFIle)
            pred.load_model()
            result = pred.makepredictions()
            flash(result)
            return redirect(url_for('index'))
        elif audioFIle and not allowed_file(audioFIle.filename):
            flash("Only .wav files supported!")
            return redirect(url_for('index'))
        else:
            flash("No File selected!")
            return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))

if __name__=='__main__':
    app.run(threaded=True,debug=True)

