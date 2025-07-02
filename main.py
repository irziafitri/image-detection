import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LOAD MODEL
model = load_model(os.path.join(BASE_DIR, 'model', 'best_model.h5'))

# CEK EKSTENSI FILE
def is_allowed_file(filename):
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions

# KONVERSI KE JPG
def convert_to_jpg(image_path):
    filename = os.path.basename(image_path)
    if filename.lower().endswith(('.png', '.jpeg', 'JPG')):
        img = Image.open(image_path).convert('RGB')
        new_filename = f"{os.path.splitext(filename)[0]}.jpg"
        new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        img.save(new_path, 'JPEG', quality=95)
        os.remove(image_path) 
        return new_filename
    return filename

# RESIZE JIKA BESAR
def resize_if_large(image_path, max_size=8000):
    img = Image.open(image_path)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        img.save(image_path)

# ERROR LEVEL ANALYSIS
def perform_ela(image_path, quality=90):
    original_filename = os.path.basename(image_path)
    converted_filename = convert_to_jpg(image_path)
    converted_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)

    temp_filename = converted_path.replace(".jpg", "_temp.jpg")
    ela_filename = converted_path.replace(".jpg", "_ela.jpg")

    image = Image.open(converted_path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if any(ex[1] for ex in extrema) else 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_image.save(ela_filename)
    os.remove(temp_filename)

    return ela_filename, original_filename, converted_filename

# KLASIFIKASI GAMBAR
def classify_image_with_model(image_path):
    ela_path, original_filename, saved_filename = perform_ela(image_path)
    img = Image.open(ela_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # PREDIKSI
    pred = model.predict(img_array)
    confidence = float(pred[0][0])
    label = 'Asli' if confidence > 0.5 else 'Manipulasi'

    return label, confidence, os.path.basename(ela_path), original_filename, saved_filename

# ROUTES 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('check.html', error="Harap pilih file!")

        file = request.files['image']
        if file.filename == '':
            return render_template('check.html', error="Nama file kosong!")

        if not is_allowed_file(file.filename):
            return render_template('check.html', error="Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.")

        try:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("Path untuk simpan:", path)

            file.save(path)
            print("File berhasil disimpan:", os.path.exists(path))

            resize_if_large(path)

            label, confidence, ela_filename, original_filename, saved_filename = classify_image_with_model(path)

            return render_template('check.html',
                                   result=label,
                                   confidence=confidence,
                                   original_filename=original_filename,
                                   saved_filename=saved_filename,
                                   ela_filename=ela_filename,
                                   success="File berhasil diproses!")
        except Exception as e:
            return render_template('check.html', error=f"Terjadi error: {str(e)}")

    return render_template('check.html')

if __name__ == '__main__':
    app.run(debug=True)
