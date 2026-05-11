import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import json

from util.framework import FewShotNERFramework
from util.data_loader import get_loader
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from transformers import BertTokenizer

# ====== CONFIG ======
N = 2
K = 2
Q = 2
max_length = 100
batch_size = 1
dev_path = "data/dev.txt"
test_path = "data/test.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iter = 3000


# ====== CHỈ LẤY MÔ HÌNH PROTO ======
def get_proto_model():
    path = "saved_models/proto.pt"
    if os.path.exists(path):
        return [path]
    return []


# ====== LOAD LOSS (nếu có) ======
def load_loss(prefix):
    loss_file = os.path.join("saved_models", prefix + "_loss.json")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            return json.load(f)
    return None


# ====== EVAL ======
def evaluate_model(model_path):
    print(f"\nEvaluating: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        pretrain_ckpt = checkpoint.get('pretrain_ckpt', "bert-base-uncased")
        model_type = checkpoint.get('model_type', 'proto')
        saved_opt = checkpoint.get('opt', {})
    else:
        state = checkpoint
        pretrain_ckpt = "bert-base-uncased"
        model_type = 'proto'
        saved_opt = {}

    # ===== tokenizer + encoder =====
    if 'roberta' in pretrain_ckpt.lower():
        tokenizer = RobertaTokenizer.from_pretrained(pretrain_ckpt)
        encoder = RobertaWordEncoder(pretrain_ckpt)
    else:
        tokenizer = BertTokenizer.from_pretrained(pretrain_ckpt)
        encoder = BERTWordEncoder(pretrain_ckpt)

    use_N = int(saved_opt.get('N', N))
    use_K = int(saved_opt.get('K', K))
    use_Q = int(saved_opt.get('Q', Q))

    val_loader = get_loader(dev_path, tokenizer, use_N, use_K, use_Q, batch_size, max_length, use_sampled_data=False)
    test_loader = get_loader(test_path, tokenizer, use_N, use_K, use_Q, batch_size, max_length, use_sampled_data=False)

    framework = FewShotNERFramework(None, val_loader, test_loader, use_sampled_data=False)

    # ===== model =====
    if model_type == 'proto':
        model = Proto(encoder)
    else:
        return None

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    precision, recall, f1, _, _, _, _ = framework.eval(model, eval_iter, ckpt='none')

    return precision, recall, f1


model_paths = get_proto_model()
if not model_paths:
    print("Không tìm thấy file saved_models/proto.pt. Vui lòng kiểm tra lại.")
    exit()

names = []
precision_list = []
recall_list = []
f1_list = []

loss_curves = {}

for path in model_paths:
    name = os.path.basename(path).replace(".pt", "").replace(".pth", "")
    result = evaluate_model(path)

    if result is None:
        continue

    p, r, f1 = result

    names.append(name)
    precision_list.append(p)
    recall_list.append(r)
    f1_list.append(f1)

    # load loss nếu có
    loss_data = load_loss(name)
    if loss_data:
        loss_curves[name] = loss_data


# ====== PLOT METRICS ======
x = np.arange(len(names))
width = 0.2

plt.figure(figsize=(10,6)) # Tăng chiều cao biểu đồ một chút cho thoáng

# Vẽ cột và gán vào các biến
bars1 = plt.bar(x - width, precision_list, width, label='Precision', color='#1f77b4')
bars2 = plt.bar(x, recall_list, width, label='Recall', color='#ff7f0e')
bars3 = plt.bar(x + width, f1_list, width, label='F1-score', color='#2ca02c')

# Hàm phụ để tự động gắn số lên đỉnh cột
def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        # Ghi số với 4 chữ số thập phân, căn giữa, in đậm
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                 f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Gọi hàm để gắn số cho cả 3 nhóm cột
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.xticks(x, names, rotation=30)
plt.ylabel("Score")
plt.ylim(0, 1.15) # Nới rộng trục Y lên 1.15 để có không gian hiển thị số ở trên cùng
plt.title("Hiệu năng mô hình Proto (Tập Test)")

plt.legend(loc='lower right') # Đưa chú thích xuống góc dưới cho khỏi đè lên cột
plt.grid(axis='y', linestyle='--', alpha=0.7) # Vẽ lưới ngang cho dễ nhìn
plt.tight_layout() # Tự động căn chỉnh lề không bị cắt chữ
plt.show()
