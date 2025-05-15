import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import ValidationError, InputRequired
from flask_wtf.file import FileAllowed 


app= Flask(__name__)
app.config['SECRET_KEY'] ='dev'
app.config['UPLOAD_FOLDER'] = 'static/uploads'


class UploadImage(FlaskForm):   #using FlaskForm to create the html form and validate the input 
    file = FileField( validators=[InputRequired(), FileAllowed(['jpg','png','jpeg'])])
    submit = SubmitField('Upload')

@app.route('/', methods=["POST","GET"])
def index():
    form = UploadImage()
    if form.validate_on_submit():
        file = form.file.data #first grab the file 
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        return "file has been uploaded"
    return render_template('upload.html', form = form)





if __name__=='__main__':
    app.run(debug=True)