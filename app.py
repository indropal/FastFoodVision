from flask import *  
import cv2
import numpy as np
from food101 import fastFood101Modellabels, fastFood101Model
import base64
import tensorflow as tf
import os

modelWeightPath ="C:/Users/palin/Desktop/[ML_DL]Projects_Study/[PROJECT] FOOD101/[GitHub Commit]fastFoodVision/food101_finetune_3.ckpt"

app = Flask(__name__)  

#fast food image inference
def fastFoodInference( npImage ):

    # instantiate the model & load saved weights
    fastfood101Model = fastFood101Model( )
    # load model weights 
    fastfood101Model.load_weights(modelWeightPath)
    # initiate prediction
    fastfoodInf = fastfood101Model.predict( npImage )[0]
    fastfoodInf =  np.argmax( fastfoodInf, axis = 0 )
    
    return fastFood101Modellabels()[fastfoodInf]


@app.route('/')  
def upload():  
    return render_template("index.html")  


@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        """
            'f' is of data type: <class 'werkzeug.datastructures.FileStorage'>
            We can change this file type to a cv2 image as follows:
            - from 'requests.files['file']' : read the string data by using '.read()'
            - convert the string data to numpy array
            - then finally convert the numpy array to an image 
        """
        if f.filename.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
            # check if the file is in 'jpg', 'jpeg' or 'png' format if not then show error message
             return """
                    <html>
                        <head>
                            <meta charset = "utf-8">
                            <title>Wrong File Type</title>
                            <!--Bootstrap-->
                            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                        </head>
                        <body style="text-align: center; margin-top: 10%; background-image: url( '/static/img/background.jpg' )";>
                            <h2>Are you sure you've uploaded the right file ?<br>The file seems to be corrupt or is not of type '.jpeg', '.jpg', or '.png' <br><br> Please check & try again ! <br> 🙂 <br><br></h2>
                            <a href = "http://127.0.0.1:5000/"><h2 class="btn btn-danger btn-lg">Home</h2></a>
                        </body>
                    </html>
                    """

        filestr = f.read()
        npimg = np.frombuffer(filestr, np.uint8) # or fromstring(filestr, np.uint8)

        # Now, convert numpy-array from the byte string to an image using cv2
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        viewImage =  cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)[:-1]
        _, viewImage = cv2.imencode('.png', viewImage)
        viewImage = base64.b64encode( viewImage ).decode('utf-8')

        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_NEAREST );

        # reshape to appropriate shape for running inference
        img = img.reshape( (1, 224, 224, 3) )

        # pass for inference from the FastFood101 model
        prediction = fastFoodInference( img )

        return render_template( "inference.html",
                                image_data = viewImage,
                                Model_predict = prediction.replace('_', ' ')
                              )


if __name__ == '__main__':  
    app.run(debug = True) 