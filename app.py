from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)
loaded_model=joblib.load('All_models.pkl')
algorithms=list(loaded_model.keys())


@app.route('/')
def home():  # put application's code here
    return  render_template('Web.html',algorithms=algorithms)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        algorithm_key=request.form['algorithm']
        model=loaded_model[algorithm_key]
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])


    print("Available keys in loaded_model:", list(loaded_model.keys()))
    prediction=model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                      compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                      fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                                      smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                                      fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
                                      smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                                      symmetry_worst, fractal_dimension_worst]])
    # prediction2=model2.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    #                                   compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    #                                   fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
    #                                   smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
    #                                   fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
    #                                   smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
    #                                   symmetry_worst, fractal_dimension_worst]])
    # prediction3=model3.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    #                                   compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    #                                   fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
    #                                   smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
    #                                   fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
    #                                   smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
    #                                   symmetry_worst, fractal_dimension_worst]])
    diagnosis = 'Malignant' if prediction== 1 else 'Benign'


    return render_template('results.html',diagnosis=diagnosis)


if __name__== '__main__':
    app.run(debug=True)
