from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/#form",methods=["GET","POST"])
def predict_data():
    
    if request.method=="POST":
        data = CustomData(
            carat = float(request.form.get("carat")),
            depth = float(request.form.get("depth")),
            table = float(request.form.get("table")),
            x = float(request.form.get("x")),
            y = float(request.form.get("y")),
            z = float(request.form.get("z")),
            cut = request.form.get("cut"),
            color = request.form.get("color"),
            clarity = request.form.get("clarity"),
        )
        final_data = data.get_dataframe()
        predict_pipe = PredictPipeline()
        pred = predict_pipe.predict(final_data)
        results = round(pred[0],2)

        return render_template("index.html",final_result=results)
    else:
        render_template("index.html")

       

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)