from flask import Flask,request,jsonify
import pickle
import numpy as np
app = Flask(__name__)

pickled_model = pickle.load(open('model.pkl','rb'))

@app.route('/',methods=["GET"])
def test():
    return jsonify({'message' : 'it works!'})

@app.route('/movies',methods=["POST"])
def getData():
    """ director = {'director': request.get_json['director']}
    actor1 = {'actor1': request.get_json['actor1']}
    actor2 = {'actor2': request.get_json['actor2']}
    genre = {'genre': request.get_json['genre']}
    budget = {'budget': request.get_json['budget']} """

    data = request.get_json(force=True)
    
    prediction = pickled_model.predict([np.array(data["exp1","exp2","exp3","exp4","exp5"])])
    return jsonify(prediction)
    

if __name__ == '__main__':
    app.run(port=8080,debug=True)
