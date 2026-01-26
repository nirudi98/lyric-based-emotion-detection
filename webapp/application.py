from flask import Flask, render_template, request
# from models import model_1_predict, model_2_predict, model_3_predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        lyrics = request.form["lyrics"]
        selected_model = request.form["model"]

        # dummy output for now
        prediction = f"dummy emotion output from {selected_model}"

        # if selected_model == "model1":
        #     prediction = model_1_predict(lyrics)
        # elif selected_model == "model2":
        #     prediction = model_2_predict(lyrics)
        # elif selected_model == "model3":
        #     prediction = model_3_predict(lyrics)
        # else:
        #     prediction = "No model selected"

        return render_template(
            "result.html",
            lyrics=lyrics,
            model=selected_model,
            prediction=prediction
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
