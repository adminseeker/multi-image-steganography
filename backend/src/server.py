from flask import Flask, request
from flask_cors import CORS
from uuid import uuid4 as uuid
import os
import script
import base64


app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return {"msg": "Made with pain by adminseeker && Bhanu Prakash"}


@app.route('/upload', methods=["POST"])
def upload():
    if request.method=="POST":
        fid=str(uuid())
        os.mkdir('./inputs/'+fid)
        f = request.get_json()
        with open("./inputs/"+fid+"/container.jpeg", "wb") as fh:
            fh.write(f['container'].decode('base64'))
        with open("./inputs/"+fid+"/input1.jpeg", "wb") as fh:
            fh.write(f['input1'].decode('base64'))
        with open("./inputs/"+fid+"/input2.jpeg", "wb") as fh:
            fh.write(f['input2'].decode('base64'))
        with open("./inputs/"+fid+"/input3.jpeg", "wb") as fh:
            fh.write(f['input3'].decode('base64'))
        script.main(fid)
        return {"msg": "Uploaded images successfully!!!","fid":fid }

@app.route('/encode', methods=["POST"])
def encode():
    data=request.get_json()
    fid=data['fid']
    original=''
    encoded=''
    try:
        with open("./outputs/"+fid+"/container.jpeg", "rb") as image_file:
            original = base64.b64encode(image_file.read())            
        with open("./outputs/"+fid+"/encoded.jpeg", "rb") as image_file:
            encoded = base64.b64encode(image_file.read())
        return{"original":original,"encoded":encoded}
    except:
        return {"msg":"No Files uploaded by you!"}

@app.route('/decode', methods=["POST"])
def decode():
    data=request.get_json()
    fid=data['fid']
    decoded1=''
    decoded2=''
    decoded3=''
    secret1=''
    secret2=''
    secret3=''
    try:
        with open("./outputs/"+fid+"/decoded1.jpeg", "rb") as image_file:
            decoded1 = base64.b64encode(image_file.read())
        with open("./outputs/"+fid+"/decoded2.jpeg", "rb") as image_file:
            decoded2 = base64.b64encode(image_file.read())
        with open("./outputs/"+fid+"/decoded3.jpeg", "rb") as image_file:
            decoded3 = base64.b64encode(image_file.read())                
        
        with open("./outputs/"+fid+"/secret1.jpeg", "rb") as image_file:
            secret1 = base64.b64encode(image_file.read()) 
        with open("./outputs/"+fid+"/secret2.jpeg", "rb") as image_file:
            secret2 = base64.b64encode(image_file.read())
        with open("./outputs/"+fid+"/secret3.jpeg", "rb") as image_file:
            secret3 = base64.b64encode(image_file.read())  
        return{"decoded1":decoded1,"decoded2":decoded2,"decoded3":decoded3,"secret1":secret1,"secret2":secret2,"secret3":secret3}
    except:
        return {"msg":"No Files uploaded by you!"}

if __name__ == "__main__":
    app.run(debug=False)
