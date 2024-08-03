import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import io, base64

from flask import Flask, request, jsonify
from flask import current_app
from flask import Flask, render_template
from flask_wtf import FlaskForm
from PIL import Image

from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

from src.findArt import segmentArt, load_bg_model, find_homographic, find_countours_and_segment_output
from src.imageSegmentor import load_sam_model
from src.find_artwork_bbox import load_yolo_model


application = Flask(__name__)
application.config['SECRET_KEY'] = 'supersecretkey'
application.config['UPLOAD_FOLDER'] = 'static/files'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@application.route('/', methods=['GET',"POST"])
@application.route('/segmentart', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file_save = os.path.join(os.path.abspath(os.path.dirname(__file__)),application.config['UPLOAD_FOLDER'])
        os.makedirs(file_save, exist_ok=True)
        os.makedirs('static/img', exist_ok=True)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),application.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        img   = cv2.imread(os.path.join(os.path.abspath(os.path.dirname(__file__)),application.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        
        # load pre-loaded model from global variable
        with application.app_context():
            mask, type_, bbox = segmentArt(img, current_app.yolo_model, current_app.sam_model, current_app.bgremoval_model)

        if type_ == "bounding_box":
            output = find_homographic(bbox, img)

        else:
            output = find_countours_and_segment_output(mask, img)

        try:
            h, w, _  = output.shape

        except:
            h, w = output.shape

        if w > 1048:
            h = int(h / 6)
            w = int(w / 6)

        cv2.imwrite('static/img/mask.png', mask)
        cv2.imwrite('static/img/output.png', output)
        return render_template('output.html', width=w, height=h, type=type_)
        
    return render_template('index.html', form=form)

@application.route('/detectart', methods=["POST"])
def detectart():
    request_data = request.get_json()
    imgstr       = request_data['image']  

    # Convert base64 to numpy array
    img       = Image.open(io.BytesIO(base64.decodebytes(bytes(imgstr, "utf-8"))))
    img_array = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)

    # Remove background from the image
    with application.app_context():
        mask, type_, bbox = segmentArt(img_array, current_app.yolo_model, current_app.sam_model, current_app.bgremoval_model)

    if type_ == "bounding_box": output = find_homographic(bbox, img)
    else: output = find_countours_and_segment_output(mask, img)

    # Convert
    pil_img = Image.fromarray(output)
    buff    = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    img       = Image.open(io.BytesIO(base64.decodebytes(bytes(new_image_string, "utf-8"))))
    img_array = np.asarray(img)
   
    if type(bbox)==type(np.array([1])):
        bbox_str = f"{bbox.tolist()}"

    elif type(bbox)==type([]):
        bbox_str = f"{bbox}"

    else:
        bbox_str = bbox
    
    if bbox != 'None':
        return jsonify({'base64Image':new_image_string, "type":type_, "bbox":bbox_str})

    else:
        return jsonify({'base64Image':new_image_string, "type":type_})

if __name__ == '__main__':
    yolo_model_path = 'weights/yolo_model_ver_2.pt'
    sam_model_path  = './weights/sam_vit_h_4b8939.pth'
    model           = "isnet-general-use"

    # Load Deep Learning Models Only Once
    with application.app_context():
        current_app.sam_model       = load_sam_model(sam_model_path)
        current_app.yolo_model      = load_yolo_model(yolo_model_path)
        current_app.bgremoval_model = load_bg_model(model)

    application.run("0.0.0.0", debug=True)