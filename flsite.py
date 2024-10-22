import pickle

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [
        {"name": "Neuron", "url": "p_lab4"}]


# Загрузка весов из файла
new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron_weights.txt')
model_class = tf.keras.models.load_model('model/classification_model.h5')

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_lab4", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Болен', 'Не болен'))
        return render_template('lab4.html', title="Первый нейрон", menu=menu,
                               class_model="Это:" + str(*np.where(predictions >= 0.5, 'Болен', 'Не болен')))




@app.route('/api_class', methods=['get'])
def predict_classification():
    # Получение данных из запроса http://localhost:5000/api_class?temp=41&pulse=90&pain_level=9
    input_data = np.array([[int(request.args.get('temp')),
                       int(request.args.get('pulse')),
                       int(request.args.get('pain_level'))]])
    print(input_data)
    # input_data = np.array(input_data.reshape(-1, 1))

    # Предсказание
    predictions = model_class.predict(input_data)
    print(predictions)
    result = 'bolen' if predictions >= 0.5 else 'ne bolen'
    print(result)
    # меняем кодировку
    app.config['JSON_AS_ASCII'] = False
    return jsonify(status = str(result))

if __name__ == "__main__":
    app.run(debug=True)
