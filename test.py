import os
import json
import torch
from transformers import BertTokenizer
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from util.data_loader import get_loader

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

def main():
    model_path = 'saved_models/proto.pt'
    
    # 1. Kiểm tra xem file model có tồn tại không
    if not os.path.exists(model_path):
        print(f" ❌ Lỗi: Không tìm thấy mô hình tại {model_path}")
        print("Hãy chắc chắn bạn đã chạy file train_proto.py thành công để sinh ra mô hình.")
        return

    print("⏳ Đang tải mô hình Proto và BERT Encoder... (Có thể mất vài giây)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = BERTWordEncoder("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    model = Proto(encoder)
    
    # 2. Load mô hình
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()
    
    print(" ✅ Tải mô hình thành công!\n")
    print("="*60)
    print(" CHƯƠNG TRÌNH TEST MÔ HÌNH FEW-SHOT NER (ProtoNet)")
    print("Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("="*60)

    # 3. Với Few-Shot NER, bắt buộc phải có Support Set để định nghĩa thực thể
    # Ở đây dùng một tập support mẫu cho demo
    support_text = """Apple B-COMPANY
Inc I-COMPANY
is O
good O

New O
York B-LOCATION
is O
big O"""

    types, supp_words, supp_labels = parse_support_set(support_text)
    if not types:
        print("Lỗi: Support set không có thực thể nào được đánh nhãn.")
        return
        
    print(f"\n[ℹ] Đang sử dụng Support Set mặc định với các thực thể: {', '.join(types)}")
    print("[ℹ] (Bạn có thể đổi Support Set trong code nếu muốn test thực thể khác)")

    # 4. Vòng lặp cho phép người dùng nhập câu liên tục
    while True:
        user_input = input("\n Nhập câu truy vấn của bạn: ")
        
        # Lệnh thoát
        if user_input.strip().lower() in ['exit', 'quit']:
            print("👋 Tạm biệt!")
            break
            
        # Nếu nhập rỗng thì bỏ qua
        if not user_input.strip():
            continue

        query_words = [user_input.split()]
        query_labels = [["O"] * len(query_words[0])]
        
        # Tạo JSON đúng chuẩn mà FewShotNERDataset cần
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
        
        with open("temp_test.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            
        # 5. Dùng get_loader để tokenize và tạo batch
        loader = get_loader(
            "temp_test.json",
            tokenizer,
            N=len(types), K=1, Q=1,
            batch_size=1,
            max_length=100,
            use_sampled_data=True
        )

        print("\n Kết quả phân tích:")
        
        found_entities = False
        with torch.no_grad():
            for batch_support, batch_query in loader:
                if torch.cuda.is_available():
                    batch_support = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch_support.items()}
                    batch_query = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch_query.items()}
                    
                # 6. Đưa vào mô hình để dự đoán
                logits, pred = model(batch_support, batch_query) 
                preds = pred.cpu().numpy()
                
                label2tag = batch_query['label2tag'][0]
                
                tokens = []
                for w in query_words[0]:
                    tokens.extend(tokenizer.tokenize(w))
                
                # 7. In kết quả ra màn hình
                for t, p in zip(tokens, preds):
                    tag = label2tag.get(p, "O")
                    if t.startswith("##"):
                        t = t.replace("##", "")
                    if tag != "O":
                        print(f"  • {t: <20} | Nhãn: {tag: <10}")
                        found_entities = True
                break
                
        if not found_entities:
            print("  -> Mô hình không tìm thấy thực thể nào trong câu này (Toàn nhãn 'O').")

if __name__ == "__main__":
    main()