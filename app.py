import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

# Vercel looks exactly for a top-level variable named 'app'
app = Flask(__name__)

# Load your pickled AdaBoostClassifier model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pkl")
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Single-file HTML template styled with Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DevPredict AI Model</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-900 text-slate-100 font-sans min-h-screen flex items-center justify-center p-4 md:p-8">

    <div class="max-w-5xl w-full bg-slate-800 rounded-3xl shadow-2xl overflow-hidden grid grid-cols-1 lg:grid-cols-12 border border-slate-700">
        
        <div class="lg:col-span-4 bg-gradient-to-br from-indigo-600 to-purple-800 p-8 flex flex-col justify-between text-white">
            <div>
                <h1 class="text-3xl font-black tracking-tight mb-3">DevPredict AI</h1>
                <p class="text-indigo-200 text-sm leading-relaxed mb-6">
                    Inference engine built on top of your trained AdaBoost classification model. Fill workspace telemetry parameters to calculate real-time developer outcomes.
                </p>
            </div>
            
            <div class="mt-6 bg-slate-950/40 backdrop-blur-md rounded-2xl p-6 border border-white/10 shadow-inner">
                <span class="text-xs font-bold uppercase tracking-widest text-indigo-300 block mb-2">Model Prediction</span>
                {% if prediction is not none %}
                    <div class="text-center py-4">
                        <span class="text-xs text-indigo-200 block mb-1">Class Assignment Outcome</span>
                        <div class="text-5xl font-black text-white tracking-tight drop-shadow-md">{{ prediction }}</div>
                    </div>
                {% else %}
                    <div class="text-center py-6 text-indigo-200 italic text-sm">
                        Waiting for metrics submission...
                    </div>
                {% endif %}
            </div>
        </div>

        <form method="POST" action="/predict" class="lg:col-span-8 p-6 md:p-10 grid grid-cols-1 sm:grid-cols-2 gap-5 bg-slate-800">
            <div class="sm:col-span-2 border-b border-slate-700 pb-2 mb-2">
                <h2 class="text-xl font-bold text-white">Developer Analytics & Cognitive Metrics</h2>
                <p class="text-slate-400 text-xs mt-1">Provide all 9 system feature weights to run the vector simulation.</p>
            </div>
            
            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Hours Coding</label>
                <input type="number" step="0.1" name="Hours_Coding" required value="{{ inputs.get('Hours_Coding', '6.5') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">AI Usage Hours</label>
                <input type="number" step="0.1" name="AI_Usage_Hours" required value="{{ inputs.get('AI_Usage_Hours', '2.5') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Lines of Code</label>
                <input type="number" name="Lines_of_Code" required value="{{ inputs.get('Lines_of_Code', '320') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Commits</label>
                <input type="number" name="Commits" required value="{{ inputs.get('Commits', '5') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Bugs Reported</label>
                <input type="number" name="Bugs_Reported" required value="{{ inputs.get('Bugs_Reported', '2') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Sleep Hours</label>
                <input type="number" step="0.1" name="Sleep_Hours" required value="{{ inputs.get('Sleep_Hours', '7.0') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Distractions</label>
                <input type="number" name="Distractions" required value="{{ inputs.get('Distractions', '2') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div>
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Cognitive Load</label>
                <input type="number" step="0.1" name="Cognitive_Load" required value="{{ inputs.get('Cognitive_Load', '3.8') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div class="sm:col-span-2">
                <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Stress Level</label>
                <input type="number" step="0.1" name="Stress_Level" required value="{{ inputs.get('Stress_Level', '4.2') }}"
                       class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
            </div>

            <div class="sm:col-span-2 mt-2">
                <button type="submit" 
                        class="w-full py-3.5 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl shadow-lg transition duration-200 cursor-pointer text-center tracking-wide uppercase text-sm">
                    Evaluate Performance Metrics
                </button>
            </div>
        </form>

    </div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None, inputs={})

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template_string(HTML_TEMPLATE, prediction="Missing Binary File Error", inputs=request.form)

    try:
        features = [
            float(request.form.get("Hours_Coding", 0)),
            float(request.form.get("AI_Usage_Hours", 0)),
            float(request.form.get("Lines_of_Code", 0)),
            float(request.form.get("Commits", 0)),
            float(request.form.get("Bugs_Reported", 0)),
            float(request.form.get("Sleep_Hours", 0)),
            float(request.form.get("Distractions", 0)),
            float(request.form.get("Cognitive_Load", 0)),
            float(request.form.get("Stress_Level", 0))
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=str(prediction), inputs=request.form)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Execution Failed: {str(e)}", inputs=request.form)

if __name__ == "__main__":
    app.run(debug=True)
