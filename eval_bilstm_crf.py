from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
import matplotlib.pyplot as plt
import os
import re

def evaluate_and_plot():
    print("=== ĐANG NẠP MÔ HÌNH VÀ DỮ LIỆU TEST ===")
    # 1. Nạp file dữ liệu test (Chú ý cột 3 là nhãn NER)
    columns = {0: 'text', 3: 'ner'}
    corpus = ColumnCorpus('data', columns, test_file='test.txt')

    # 2. Nạp file mô hình đã lưu
    model_path = 'saved_models/bilstm_crf.pt'
    
    try:
        model = SequenceTagger.load(model_path)
        print(f" Đã load mô hình từ: {model_path}")
    except Exception as e:
        print(f"Không thể load mô hình. Hãy chắc chắn bạn đã chạy file train trước. Lỗi: {e}")
        return

    print("\n=== KẾT QUẢ ĐÁNH GIÁ (TEST SET) ===")
    # 3. Tiến hành đánh giá
    result = model.evaluate(corpus.test, gold_label_type='ner', mini_batch_size=32)


    print("\n=== ĐANG VẼ BIỂU ĐỒ ===")
    try:
        # 4. Tự động đọc điểm số Precision, Recall, F1 từ chuỗi văn bản của Flair
        # Flair thường in tổng kết ở dòng có chữ "micro avg" hoặc "avg / total"
        match = re.search(r'(?:micro avg|avg / total)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', result.detailed_results)
        
        if match:
            precision = float(match.group(1))
            recall = float(match.group(2))
            f1 = float(match.group(3))
            
            print(f"Dữ liệu vẽ biểu đồ: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            # 5. Cấu hình và hiển thị biểu đồ Bar Chart
            metrics = ['Precision', 'Recall', 'F1-Score']
            scores = [precision, recall, f1]

            plt.figure(figsize=(8, 6))
            bars = plt.bar(metrics, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.4)

            # Gắn số liệu trực tiếp lên đầu mỗi cột
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                         f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

            # Làm đẹp biểu đồ
            plt.ylim(0, 1.1) # Giới hạn trục Y từ 0 đến 1.1 (để có khoảng trống ghi số)
            plt.ylabel('Điểm số (Score)')
            plt.title('Hiệu năng mô hình BiLSTM-CRF (Tập Test)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Hiển thị
            plt.show()
        else:
            print("Không thể tự động trích xuất điểm số để vẽ biểu đồ từ log của Flair.")
            
    except Exception as e:
        print(f" Có lỗi xảy ra trong quá trình vẽ biểu đồ: {e}")

if __name__ == "__main__":
    evaluate_and_plot()