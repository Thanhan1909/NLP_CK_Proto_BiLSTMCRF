import torch
import matplotlib.pyplot as plt
import numpy as np

from util.framework import FewShotNERFramework
from util.data_loader import get_loader
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from model.nnshot import NNShot
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

# ====== LOAD TOKENIZER (FIX LỖI) ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ====== LOAD DATA ======
val_loader = get_loader(
    dev_path,
    tokenizer,
    N, K, Q,
    batch_size,
    max_length,
    use_sampled_data=False
)
test_loader = get_loader(
    test_path,
    tokenizer,   
    N, K, Q,
    batch_size,
    max_length,
    use_sampled_data=False
)

# ====== FRAMEWORK ======
framework = FewShotNERFramework(
    None, val_loader, test_loader,
    use_sampled_data=False
)

# ====== FUNCTION EVAL ======
def evaluate_model(model_type, model_path):
    print(f"\nEvaluating {model_type}...")

    # tạo encoder mới mỗi lần
    encoder = BERTWordEncoder("bert-base-uncased")

    if model_type == "proto":
        model = Proto(encoder)
    elif model_type == "nnshot":
        model = NNShot(encoder)
    else:
        raise ValueError("Unknown model")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    precision, recall, f1, _, _, _, _ = framework.eval(model, 500)

    print(f"{model_type} -> P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1


# ====== LOAD & EVAL ======
proto_p, proto_r, proto_f1 = evaluate_model(
    "proto", "saved_models/proto.pt"
)

nn_p, nn_r, nn_f1 = evaluate_model(
    "nnshot", "saved_models/nnshot.pt"
)

# ====== PLOT ======
models = ['ProtoNet', 'NNShot']
precision = [proto_p, nn_p]
recall = [proto_r, nn_r]
f1 = [proto_f1, nn_f1]

x = np.arange(len(models))
width = 0.2

plt.figure()

plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-score')

plt.xticks(x, models)
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Few-shot NER Comparison")

plt.legend()
plt.grid()

plt.show()