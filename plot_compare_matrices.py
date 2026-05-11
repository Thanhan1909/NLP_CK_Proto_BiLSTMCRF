import os
import torch
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Thư viện cho BiLSTM-CRF ===
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.data import Sentence

# === Thư viện cho ProtoNet ===
from transformers import BertTokenizer
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from util.data_loader import get_loader

# Tắt các cảnh báo đỏ cho Terminal sạch sẽ
warnings.filterwarnings("ignore")

def clean_tag(tag):
    """ Hàm gộp các nhãn B- và I- thành nhãn gốc (PER, ORG, LOC, MISC) cho dễ nhìn """
    if not tag or tag == "O": return "O"
    return tag.replace("B-", "").replace("I-", "")

def get_bilstm_predictions():
    print("=== ĐANG NẠP MÔ HÌNH BILSTM-CRF ===")
    columns = {0: 'text', 3: 'ner'}
    corpus = ColumnCorpus('data', columns, test_file='test.txt') 
    model = SequenceTagger.load('saved_models/bilstm_crf.pt')

    y_true = []
    y_pred = []

    print("Đang dự đoán BiLSTM-CRF (Quét toàn bộ tập test)...")
    for sentence in corpus.test:
        words = []
        for token in sentence:
            words.append(token.text)
            gold = "O"
            if hasattr(token, 'get_labels'):
                labels = token.get_labels('ner')
                if labels: gold = labels[0].value
            elif hasattr(token, 'get_label'):
                label = token.get_label('ner')
                if label: gold = label.value
            y_true.append(clean_tag(gold))

        clean_sentence = Sentence(words)
        model.predict(clean_sentence)

        for token in clean_sentence:
            pred = "O"
            if hasattr(token, 'get_labels'):
                labels = token.get_labels('ner')
                if labels: pred = labels[0].value
            elif hasattr(token, 'get_label'):
                label = token.get_label('ner')
                if label: pred = label.value
            y_pred.append(clean_tag(pred))

    return y_true, y_pred

def get_proto_predictions():
    print("=== ĐANG NẠP MÔ HÌNH PROTONET ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BERTWordEncoder("bert-base-uncased")
    model = Proto(encoder).to(device)

    # Load checkpoint
    ckpt = torch.load('saved_models/proto.pt', map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Tạo DataLoader (Bốc thăm ngẫu nhiên 500 bài thi - 2-way 2-shot)
    test_loader = get_loader('data/test.txt', tokenizer, N=2, K=2, Q=2, batch_size=1, max_length=100, use_sampled_data=False)

    y_true = []
    y_pred = []

    print("Đang dự đoán ProtoNet (Đang làm 500 bài test ngẫu nhiên)...")
    eval_iter = 500
    with torch.no_grad():
        for i, (batch_support, batch_query) in enumerate(test_loader):
            if i >= eval_iter:
                break
            if torch.cuda.is_available():
                batch_support = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch_support.items()}
                batch_query = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch_query.items()}

            logits, pred = model(batch_support, batch_query)
            
            # Dùng flatten() để ép phẳng mảng, chống lỗi chiều dữ liệu (dimension)
            preds = pred.cpu().numpy().flatten()
            true_indices = batch_query['label'][0].cpu().numpy().flatten()
            
            # Label2tag có thể nằm trong list hoặc dict, lấy an toàn
            label2tag = batch_query['label2tag']
            if isinstance(label2tag, list):
                label2tag = label2tag[0]

            for t_idx, p_idx in zip(true_indices, preds):
                # Loại bỏ các token padding đệm thêm của BERT (thường mang nhãn -1)
                if t_idx != -1 and t_idx in label2tag:
                    t_tag = label2tag[t_idx]
                    p_tag = label2tag.get(p_idx, "O")
                    
                    y_true.append(clean_tag(t_tag))
                    y_pred.append(clean_tag(p_tag))

    return y_true, y_pred

def plot_both():
    # 1. Lấy kết quả dự đoán của 2 mô hình
    y_true_bi, y_pred_bi = get_bilstm_predictions()
    y_true_pr, y_pred_pr = get_proto_predictions()

    # 2. Tạo danh sách nhãn chung (PER, ORG, LOC, MISC, O)
    all_labels = sorted(list(set(y_true_bi + y_pred_bi + y_true_pr + y_pred_pr)))
    if 'O' in all_labels:
        all_labels.remove('O')
        all_labels.append('O') # Ép nhãn 'O' nằm ở cuối cùng cho biểu đồ đẹp hơn

    # 3. Tính toán Ma trận
    cm_bi = confusion_matrix(y_true_bi, y_pred_bi, labels=all_labels)
    cm_pr = confusion_matrix(y_true_pr, y_pred_pr, labels=all_labels)

    # 4. Vẽ biểu đồ (1 hàng, 2 cột)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Biểu đồ 1: BiLSTM-CRF (Màu Xanh dương)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_bi, display_labels=all_labels)
    disp1.plot(ax=axes[0], cmap='Blues', xticks_rotation=45, values_format='d', colorbar=False)
    axes[0].set_title('Ma trận Nhầm lẫn: BiLSTM-CRF\n(Học 100% dữ liệu)', fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('Nhãn Dự Đoán (Predicted)', fontsize=12)
    axes[0].set_ylabel('Nhãn Thực Tế (True)', fontsize=12)

    # Biểu đồ 2: ProtoNet (Màu Xanh lá)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_pr, display_labels=all_labels)
    disp2.plot(ax=axes[1], cmap='Greens', xticks_rotation=45, values_format='d', colorbar=False)
    axes[1].set_title('Ma trận Nhầm lẫn: ProtoNet\n(Học Few-Shot 2-mẫu)', fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel('Nhãn Dự Đoán (Predicted)', fontsize=12)
    axes[1].set_ylabel('Nhãn Thực Tế (True)', fontsize=12)

    # Thêm tiêu đề tổng
    fig.suptitle('SO SÁNH MỨC ĐỘ NHẬN DIỆN THỰC THỂ: TRUYỀN THỐNG vs FEW-SHOT', fontsize=20, fontweight='bold', color='#2c3e50')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Căn chỉnh lề để không đè chữ
    plt.show()

if __name__ == "__main__":
    plot_both()