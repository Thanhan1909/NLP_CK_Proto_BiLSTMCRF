import os
import shutil
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import flair
import torch

# Thiết lập sử dụng GPU cho Flair
flair.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"=== ĐANG SỬ DỤNG THIẾT BỊ: {flair.device} ===")
    print("=== ĐANG NẠP DỮ LIỆU ===")
    # Khai báo định dạng file CoNLL (Cột 0: Từ, Cột 1: Nhãn)
    columns = {0: 'text', 3: 'ner'}
    
    # Trỏ tới thư mục chứa train.txt, dev.txt, test.txt
    data_folder = 'data' 

    corpus: Corpus = ColumnCorpus(
        data_folder,
        columns,
        train_file='train.txt',
        test_file='test.txt',
        dev_file='dev.txt'
    )
    print(corpus)

    print("\n=== ĐANG KHỞI TẠO MÔ HÌNH ===")
    # Dùng GloVe embeddings cho tiếng Anh (nhẹ và hiệu quả cho Baseline)
    embedding_types = [WordEmbeddings('glove')]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # Tạo từ điển nhãn
    label_dictionary = corpus.make_label_dictionary(label_type='ner')

    # Khởi tạo BiLSTM-CRF
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dictionary,
        tag_type='ner',
        use_crf=True
    )

    print("\n=== BẮT ĐẦU HUẤN LUYỆN ===")
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # Lưu tạm vào thư mục temp_bilstm
    temp_dir = 'saved_models/temp_bilstm'
    trainer.train(
        temp_dir,
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=20 # Bạn có thể tăng lên 50 nếu muốn model hội tụ sâu hơn
    )

    print("\n=== LƯU MÔ HÌNH ===")
    # Đổi tên và di chuyển file best-model.pt ra ngoài cho giống Proto
    best_model_path = os.path.join(temp_dir, 'best-model.pt')
    final_save_path = 'saved_models/bilstm_crf.pt'
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_save_path)
        print(f"Đã lưu mô hình thành công tại: {final_save_path}")
        
        # Dọn dẹp thư mục rác
        shutil.rmtree(temp_dir)
    else:
        print("Lỗi: Không tìm thấy file model sau khi train.")

if __name__ == "__main__":
    # Đảm bảo thư mục saved_models tồn tại
    os.makedirs("saved_models", exist_ok=True)
    main()