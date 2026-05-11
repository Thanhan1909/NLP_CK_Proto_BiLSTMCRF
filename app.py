import os
import sys
import json
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import torch
from transformers import BertTokenizer
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from util.data_loader import get_loader

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ImportError:
    pass

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== INIT CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading BERT encoder...")
encoder = BERTWordEncoder("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("BERT loaded.")

def parse_support_set(text):
    words = []
    labels = []
    types = set()
    current_words = []
    current_labels = []
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            if current_words:
                words.append(current_words)
                labels.append(current_labels)
                current_words = []
                current_labels = []
            continue
        parts = line.split()
        if len(parts) >= 2:
            w = parts[0]
            l = parts[-1]
            if l != 'O':
                l = l.replace('B-', '').replace('I-', '')
            current_words.append(w)
            current_labels.append(l)
            if l != 'O':
                types.add(l)
    
    if current_words:
        words.append(current_words)
        labels.append(current_labels)
        
    return list(types), words, labels

def run_inference(model_path, model_type, support_text, query_text):
    # 1. Nếu là BiLSTM-CRF (Supervised Flair Model)
    if model_type == 'bilstm_crf':
        tagger = SequenceTagger.load(model_path)
        sentence = Sentence(query_text)
        tagger.predict(sentence)
        
        result = []
        
        # Lấy danh sách các thực thể (span) từ câu
        token_to_tag = {}
        for span in sentence.get_spans('ner'):
            for i, token in enumerate(span.tokens):
                prefix = 'B-' if i == 0 else 'I-'
                token_to_tag[token] = f"{prefix}{span.tag}"
                
        for token in sentence:
            tag = token_to_tag.get(token, "O")
            result.append((token.text, tag))
            
        return result
    # 2. Nếu là ProtoNet  (Few-Shot Model)
    types, supp_words, supp_labels = parse_support_set(support_text)
    
    if not types:
        raise ValueError("Support set must contain ít nhất 1 loại entity (nhãn khác 'O').")
        
    query_words = [query_text.split()]
    query_labels = [["O"] * len(query_words[0])]
    
    # Create JSON format cho FewShotNERDataset
    data = {
        "types": types,
        "support": {
            "word": supp_words,
            "label": supp_labels
        },
        "query": {
            "word": query_words,
            "label": query_labels
        }
    }
    
    with open("temp.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
        
    loader = get_loader(
        "temp.json",
        tokenizer,
        N=len(types), K=1, Q=1,
        batch_size=1,
        max_length=100,
        use_sampled_data=True
    )
    
    # Load Model
    if model_type == 'proto':
        model = Proto(encoder)
    else:
        model = NNShot(encoder)
        
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()

    result = []
    with torch.no_grad():
        for batch_support, batch_query in loader:
            if torch.cuda.is_available():
                batch_support = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch_support.items()}
                batch_query = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch_query.items()}
                
            logits, pred = model(batch_support, batch_query) 
            preds = pred.cpu().numpy()
            
            label2tag = batch_query['label2tag'][0]
            
            tokens = []
            for w in query_words[0]:
                tokens.extend(tokenizer.tokenize(w))
            
            for t, p in zip(tokens, preds):
                tag = label2tag.get(p, "O")
                result.append((t, tag))
                
            break
            
    return result

def colorize(result):
    html = ""
    colors = ['#ef4444', '#f97316', '#8b5cf6', '#06b6d4', '#10b981', '#ec4899']
    entity_colors = {}
    
    for word, label in result:
        is_subword = word.startswith('##')
        display_word = word.replace('##', '')
        
        if is_subword and html.endswith(' '):
            html = html[:-1]
            
        if label != "O":
            entity_type = label.replace('B-', '').replace('I-', '')
            if entity_type not in entity_colors:
                entity_colors[entity_type] = colors[len(entity_colors) % len(colors)]
            
            color = entity_colors[entity_type]
            html += f'<span class="token tag-entity" style="background-color: {color}; box-shadow: 0 0 8px {color}60;">{display_word} <span class="text-xs opacity-75">[{label}]</span></span> '
        else:
            html += f'<span class="token tag-O">{display_word}</span> '
            
    return html

HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NER Deployment Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .token { display: inline-block; padding: 2px 6px; margin: 2px; border-radius: 4px; font-size: 0.9em; }
        .tag-O { background-color: transparent; }
        .tag-entity { color: white; font-weight: bold; border: 1px solid rgba(255,255,255,0.2); }
    </style>
    <script>
        function toggleSupportSet() {
            const modelType = document.getElementById('model_type').value;
            const supportDiv = document.getElementById('support_set_div');
            if (modelType === 'bilstm_crf') {
                supportDiv.style.opacity = '0.3';
                supportDiv.style.pointerEvents = 'none';
            } else {
                supportDiv.style.opacity = '1';
                supportDiv.style.pointerEvents = 'auto';
            }
        }
        window.onload = toggleSupportSet;
    </script>
</head>
<body class="min-h-screen p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-4xl font-extrabold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-emerald-400">
            NER Deployment Platform 🚀
        </h1>
        <p class="text-slate-400 mb-8">Nền tảng triển khai các mô hình Named Entity Recognition (ProtoNet & BiLSTM-CRF).</p>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Form -->
            <div class="glass p-6 rounded-xl shadow-xl border-t border-slate-700">
                <form method="post" enctype="multipart/form-data" class="space-y-5">
                    <div>
                        <label class="block text-sm font-semibold mb-2 text-indigo-300">1. Tải lên mô hình từ máy tính (.pt / .pth)</label>
                        <input type="file" name="model_file" accept=".pt,.pth" class="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-700 bg-slate-800 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                    </div>

                    <div>
                        <label class="block text-sm font-semibold mb-2 text-indigo-300">2. Loại mô hình</label>
                        <select name="model_type" id="model_type" onchange="toggleSupportSet()" class="w-full bg-slate-800 text-white rounded-lg p-3 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                            <option value="proto" {% if model_type == 'proto' %}selected{% endif %}>ProtoNet (Few-Shot)</option>
                            <option value="bilstm_crf" {% if model_type == 'bilstm_crf' %}selected{% endif %}>BiLSTM-CRF (Supervised - Flair)</option>
                        </select>
                    </div>

                    <div id="support_set_div" class="transition-opacity duration-300">
                        <div class="flex justify-between items-end mb-2">
                            <label class="block text-sm font-semibold text-indigo-300">3. Support Set (Chỉ dành cho Few-Shot ProtoNet)</label>
                        </div>
                        
                        <!-- Box Giải thích thân thiện -->
                        <div class="bg-indigo-900/40 border border-indigo-500/30 rounded-lg p-3 mb-3 text-sm text-indigo-200">
                            <strong>💡 Support Set là gì?</strong><br>
                            ProtoNet là mô hình "Học ít mẫu" (Few-Shot). Nó không biết trước các thực thể. Bạn cần cung cấp vài câu ví dụ (Support Set) có đánh dấu từ khóa để mô hình "học lỏm" tại chỗ, sau đó nó sẽ tự đi tìm các từ tương tự trong câu truy vấn của bạn.
                        </div>

                        <!-- Các nút Preset -->
                        <div class="flex flex-wrap gap-2 mb-3 items-center">
                            <span class="text-xs text-slate-400 mr-1">Tải mẫu nhanh:</span>
                            <button type="button" onclick="loadPreset('tech')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Công nghệ</button>
                            <button type="button" onclick="loadPreset('medical')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Y tế</button>
                            <button type="button" onclick="loadPreset('sports')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Thể thao</button>
                            <button type="button" onclick="loadPreset('crypto')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Tiền ảo</button>
                            <button type="button" onclick="loadPreset('location')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Địa lý</button>
                            <button type="button" onclick="loadPreset('movies')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Điện ảnh</button>
                            <button type="button" onclick="loadPreset('animals')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Động vật</button>
                            <button type="button" onclick="loadPreset('food')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Ẩm thực</button>
                            <button type="button" onclick="loadPreset('history')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Lịch sử</button>
                            <button type="button" onclick="loadPreset('education')" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs font-medium transition border border-slate-600">Giáo dục</button>
                        </div>

                        <textarea name="support" rows="7" class="w-full bg-slate-800 text-white rounded-lg p-3 border border-slate-700 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition leading-relaxed" placeholder="Ví dụ:&#10;Apple B-COMPANY&#10;Inc I-COMPANY&#10;is O&#10;good O">{{support_text}}</textarea>
                        <p class="text-xs text-slate-400 mt-2">Định dạng CoNLL: Mỗi từ 1 dòng, cách nhãn bằng khoảng trắng. Dùng B- (Bắt đầu) và I- (Bên trong) cho cụm từ dài. Để cách 1 dòng trống giữa 2 câu.</p>
                    </div>

                    <div>
                        <label class="block text-sm font-semibold mb-2 text-indigo-300">4. Câu truy vấn (Query Text)</label>
                        <textarea name="query" rows="3" class="w-full bg-slate-800 text-white rounded-lg p-3 border border-slate-700 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition leading-relaxed" placeholder="Nhập câu cần trích xuất thực thể...">{{query_text}}</textarea>
                    </div>

                    <button type="submit" class="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-bold py-3 px-4 rounded-lg shadow-[0_0_15px_rgba(99,102,241,0.5)] transition transform hover:-translate-y-0.5 mt-4">
                        Chạy mô hình (Inference) ✨
                    </button>
                </form>
            </div>

            <!-- Result -->
            <div class="glass p-6 rounded-xl shadow-xl flex flex-col border-t border-slate-700">
                <h3 class="text-xl font-bold mb-4 text-emerald-300 border-b border-slate-700 pb-3 flex items-center">
                    <svg class="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    Kết quả Trích xuất
                </h3>
                <div class="flex-grow bg-slate-900 rounded-lg p-6 font-mono text-lg leading-loose overflow-y-auto border border-slate-700 shadow-inner">
                    {% if error %}
                        <div class="text-red-400 bg-red-900/30 p-4 rounded-lg border border-red-500/50">
                            <strong>Lỗi:</strong> {{ error }}
                        </div>
                    {% elif result_html %}
                        {{ result_html | safe }}
                    {% else %}
                        <div class="h-full flex flex-col items-center justify-center text-slate-500">
                            <svg class="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                            <p class="text-center">Tải lên mô hình và điền thông tin để xem kết quả.</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const presets = {
            'tech': {
                support: `Apple B-COMPANY
Inc I-COMPANY
makes O
iPhone B-PRODUCT
15 I-PRODUCT

Google B-COMPANY
released O
Pixel B-PRODUCT
7 I-PRODUCT`,
                query: "Microsoft announced the new Surface Pro laptop in Seattle yesterday."
            },
            'medical': {
                support: `headache B-SYMPTOM
is O
bad O

COVID-19 B-DISEASE
causes O
fever B-SYMPTOM`,
                query: "The patient has a bad cough and might be infected with Malaria."
            },
            'sports': {
                support: `LeBron B-PLAYER
James I-PLAYER
plays O
for O
Lakers B-TEAM

Lionel B-PLAYER
Messi I-PLAYER
joined O
Inter B-TEAM
Miami I-TEAM`,
                query: "Stephen Curry scored 50 points for the Golden State Warriors."
            },
            'crypto': {
                support: `Bitcoin B-COIN
is O
traded O
on O
Binance B-EXCHANGE

Ethereum B-COIN
dropped O
on O
Coinbase B-EXCHANGE`,
                query: "Solana is seeing huge volume on Kraken today."
            },
            'location': {
                support: `Tokyo B-CITY
is O
in O
Japan B-COUNTRY

Berlin B-CITY
is O
the O
capital O
of O
Germany B-COUNTRY`,
                query: "I traveled to Paris and then took a train to Italy."
            },
            'movies': {
                support: `Leonardo B-ACTOR
DiCaprio I-ACTOR
starred O
in O
Titanic B-MOVIE

Keanu B-ACTOR
Reeves I-ACTOR
was O
in O
The B-MOVIE
Matrix I-MOVIE`,
                query: "Tom Cruise plays the main role in Mission Impossible."
            },
            'animals': {
                support: `Tiger B-ANIMAL
lives O
in O
the O
Jungle B-HABITAT

Dolphin B-ANIMAL
swims O
in O
the O
Ocean B-HABITAT`,
                query: "The Penguin is a bird that survives in Antarctica."
            },
            'food': {
                support: `Sushi B-FOOD
is O
from O
Japan B-COUNTRY

Pizza B-FOOD
was O
invented O
in O
Italy B-COUNTRY`,
                query: "Pho is a very delicious and famous dish from Vietnam."
            },
            'history': {
                support: `Abraham B-PERSON
Lincoln I-PERSON
led O
during O
the O
Civil B-EVENT
War I-EVENT

Julius B-PERSON
Caesar I-PERSON
ruled O
the O
Roman B-EVENT
Empire I-EVENT`,
                query: "George Washington was a key figure in the American Revolution."
            },
            'education': {
                support: `He O
went O
to O
Harvard B-UNIVERSITY
for O
a O
Bachelor B-DEGREE

She O
studied O
at O
MIT B-UNIVERSITY
to O
get O
her O
Master B-DEGREE`,
                query: "John graduated from Stanford with a PhD."
            }
        };

        function loadPreset(key) {
            document.getElementsByName('support')[0].value = presets[key].support;
            document.getElementsByName('query')[0].value = presets[key].query;
            
            // Highlight nút vừa bấm
            const inputs = document.getElementsByName('support')[0];
            inputs.classList.add('ring-2', 'ring-emerald-500');
            setTimeout(() => inputs.classList.remove('ring-2', 'ring-emerald-500'), 500);
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    support_text = "Apple B-COMPANY\nInc I-COMPANY\nis O\na O\ngood O\ncompany O\n\nNew O\nYork B-LOCATION\nis O\nbig O"
    query_text = "Microsoft is located in Seattle"
    model_type = "proto"
    result_html = ""
    error = ""

    if request.method == "POST":
        support_text = request.form.get("support", support_text)
        query_text = request.form.get("query", query_text)
        model_type = request.form.get("model_type", "proto")
        
        if 'model_file' not in request.files:
            error = "Vui lòng chọn file mô hình."
        else:
            file = request.files['model_file']
            if file.filename == '':
                model_path = os.path.join("saved_models", f"{model_type}.pt")
                if not os.path.exists(model_path):
                    error = f"Bạn chưa upload file mô hình nào từ máy tính và không tìm thấy {model_path}!"
            else:
                filename = secure_filename(file.filename)
                model_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(model_path)
            
            if not error:
                try:
                    result = run_inference(model_path, model_type, support_text, query_text)
                    result_html = colorize(result)
                except Exception as e:
                    error = str(e)

    return render_template_string(HTML, support_text=support_text, query_text=query_text, model_type=model_type, result_html=result_html, error=error)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)