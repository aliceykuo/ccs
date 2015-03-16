# import os
# # We'll render HTML templates and access data sent by POST
# # using the request object from flask. Redirect and url_for
# # will be used to redirect the user once the upload is done
# # and send_from_directory will help us to send/show on the
# # browser the file that the user just uploaded
# from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# from werkzeug import secure_filename
# from extract_patches import ExtractPatches
# import pickle as pkl
# from collections import Counter

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/' # This is the path to the upload directory
# app.config['ALLOWED_EXTENSIONS'] = set(['zip', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']) # These are the extension that we are accepting to be uploaded

# def run_on_start(pkl_model_fname='static/rf_model.pkl'):
#     print 'Loading model'
#     return pkl.load(open(pkl_model_fname, 'rb'))

# def allowed_file(filename):
#     return '.' in filename and \
#         filename.lower().rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/testing')
# def testing():
#     return 'hello'

# @app.route('/upload', methods=['POST'])
# def upload():
#     fname_lst = []
#     file = request.files['file']
#     # Check if the file is one of the allowed types/extensions
#     if file and allowed_file(file.filename):
#         # Make the filename safe, remove unsupported chars
#         filename = secure_filename(file.filename)

#         # save file
#         print 'saving file...'
#         print filename
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         fname_lst.append(filepath)

#         # prediction
#         print 'predicting...'
#         results_page = preprocess_and_predict(fname_lst)

#     # return template prediction
#     return render_template(results_page)

# def preprocess_and_predict(fname_lst):
#     print 'Extracting Patches...'
#     ep = ExtractPatches(fname_lst)
#     x = ep.run_extract_patch()
#     print 'Feature of the image', x
#     print app.model.predict(x)
#     prediction_dict = Counter(app.model.predict(x))

#     for category, count in prediction_dict.iteritems():
#         if category == 0:
#             print 'Ballroom:', count
#         if category == 1:
#             print 'Beach:', count
#         if category == 2:
#             print 'Rustic:', count

#     most_common = prediction_dict.most_common(1)[0][0]
#     if most_common == 0:
#         return 'ballroom.html'
#     if most_common == 1:
#         return 'beach.html'
#     if most_common == 2:
#         return 'rustic.html'

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

# if __name__ == '__main__':
#     app.model = run_on_start()
#     app.run(port=5656)

import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename
from extract_patches import ExtractPatches
import pickle as pkl
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/' # This is the path to the upload directory
app.config['ALLOWED_EXTENSIONS'] = set(['zip', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']) # These are the extension that we are accepting to be uploaded

def run_on_start(pkl_model_fname='static/rf_model.pkl'):
    print 'Loading model'
    return pkl.load(open(pkl_model_fname, 'rb'))

def allowed_file(filename):
    return '.' in filename and \
        filename.lower().rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/testing')
def testing():
    return 'hello'

@app.route('/upload', methods=['POST'])
def upload():
    fname_lst = []
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)

        # save file
        print 'saving file...'
        print filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        fname_lst.append(filepath)

        # prediction
        print 'predicting...'
        results = preprocess_and_predict(fname_lst)

    # return template prediction
    # render_template('ballroom.html', predictions={'ballroom': 75, 'rustic': 25} )
    return render_template(results['template'], predictions=results['predictions'] )

def preprocess_and_predict(fname_lst):
    print 'Extracting Patches...'
    ep = ExtractPatches(fname_lst)
    x = ep.run_extract_patch()
    print 'Feature of the image', x
    print app.model.predict(x)
    prediction_dict = Counter(app.model.predict(x))

    categories = { 0: 'ballroom', 1: 'beach', 2: 'rustic'}

    results = {}
    results['predictions'] = {}
        
    # calculate percentage of results
    total_count = sum(prediction_dict.values())
    for category, count in prediction_dict.iteritems():
        # ie. theme = 'ballroom'
        theme = categories[category]
        
        # Calculate and write percentage to results 
        # ie. results = { 'predictions': {ballroom': 50, 'rustic': 25, 'beach': 25} }
        results['predictions'][theme] = (count * 100) / total_count

    # transform highest rank to template name
    most_common = prediction_dict.most_common(1)[0][0] # ie. 0
    template_theme = categories[most_common] # ie. 'ballroom'
    results['template'] = template_theme + '.html' # ie. add this to results: {'template': 'ballroom.html'}

    # ie. results = { 'predictions': {ballroom': 50, 'rustic': 25, 'beach': 25}, 'template': 'ballroom.html' }
    return results

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.model = run_on_start()
    app.run(port=5656)
