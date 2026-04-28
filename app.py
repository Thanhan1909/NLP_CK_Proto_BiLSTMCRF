from flask import Flask, request, render_template_string
import torch
from transformers import BertTokenizer
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from model.nnshot import NNShot
from util.data_loader import get_loader
from util.framework import FewShotNERFramework
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
app = Flask(__name__)

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
encoder = BERTWordEncoder("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

proto_model = Proto(encoder)
proto_model.load_state_dict(torch.load("saved_models/proto.pt", map_location=device))
proto_model.to(device)
proto_model.eval()

nnshot_model = NNShot(encoder)
nnshot_model.load_state_dict(torch.load("saved_models/nnshot.pt", map_location=device))
nnshot_model.to(device)
nnshot_model.eval()

print("✅ Models loaded!")

# ===== FAKE SUPPORT SET (đơn giản để chạy inference) =====
def build_fake_loader(sentence):
    with open("temp.txt", "w", encoding="utf-8") as f:
        for w in sentence.split():
            f.write(f"{w} O\n")
        f.write("\n")

    loader = get_loader(
        "temp.txt",
        tokenizer,
        N=2, K=1, Q=1,
        batch_size=1,
        max_length=100,
        use_sampled_data=False
    )
    return loader

# ===== INFERENCE =====
def predict(sentence, model):
    loader = build_fake_loader(sentence)
    framework = FewShotNERFramework(None, None, loader)

    result = []

    for batch in loader:
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}

        logits = model(batch)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]

        tokens = tokenizer.tokenize(sentence)

        for t, p in zip(tokens, preds):
            result.append((t, str(p)))

        break

    return result

# ===== COLOR MAP =====
def colorize(result):
    html = ""
    for word, label in result:
        if label != "O":
            html += f'<span style="background-color:yellow; padding:3px; margin:2px;">{word} ({label})</span>'
        else:
            html += f"{word} "
    return html

# ===== UI =====
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Few-shot NER Demo</title>
<style>
body { font-family: Arial; margin: 40px; }
textarea { width: 500px; height: 100px; }
button { padding: 10px; margin-top: 10px; }
.result { margin-top: 20px; }
</style>
</head>
<body>

<h2>🔥 Few-shot NER Demo</h2>

<form method="post">
    <textarea name="text" placeholder="Enter sentence...">{{text}}</textarea><br>
    
    <select name="model">
        <option value="proto">ProtoNet</option>
        <option value="nnshot">NNShot</option>
    </select>
    
    <br>
    <button type="submit">Predict</button>
</form>

<div class="result">
    {{result|safe}}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    result_html = ""

    if request.method == "POST":
        text = request.form.get("text")
        model_type = request.form.get("model")

        if model_type == "proto":
            result = predict(text, proto_model)
        else:
            result = predict(text, nnshot_model)

        result_html = colorize(result)

    return render_template_string(HTML, text=text, result=result_html)

if __name__ == "__main__":
    app.run(debug=True)