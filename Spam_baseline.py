import pandas as pd
import numpy as np
import torch
import os
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter

# --- 1. 配置和路径 ---
# 假设您的文件名为 spam.csv，包含 'v1' (label) 和 'v2' (text) 列
DATA_FILE_PATH = "spam.csv"
MAX_LEN = 150
BATCH_SIZE = 32

# 深度学习模型优化配置
LSTM_EPOCHS = 10
CNN_EPOCHS = 10
DL_LEARNING_RATE = 0.001
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100
NUM_LABELS = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 全局词汇表
VOCAB_SIZE = 0
WORD_TO_IX = {"<PAD>": 0, "<UNK>": 1}


# --- 2. 数据加载和预处理 ---
def clean_text(text):
    """针对英文文本的简单清洗"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def load_spam_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return None, None

    # Spam 数据集通常包含 latin-1 编码
    df = pd.read_csv(file_path, encoding='latin-1')
    # 仅保留前两列并重命名
    df = df.iloc[:, [0, 1]]
    df.columns = ['label', 'text']

    # 标签转数值: 'ham' -> 0, 'spam' -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(clean_text)

    return df['text'].values, df['label'].values


def build_vocab(texts):
    global VOCAB_SIZE, WORD_TO_IX
    all_words = []
    for text in texts:
        all_words.extend(text.split())

    # 过滤低频词，仅保留出现 2 次以上的词
    word_counts = Counter(all_words)
    sorted_words = [w for w, c in word_counts.items() if c > 1]

    for word in sorted_words:
        if word not in WORD_TO_IX:
            WORD_TO_IX[word] = len(WORD_TO_IX)
    VOCAB_SIZE = len(WORD_TO_IX)
    print(f"📖 词汇表构建完成，大小: {VOCAB_SIZE}")


def text_to_sequence(text):
    seq = [WORD_TO_IX.get(word, WORD_TO_IX["<UNK>"]) for word in text.split()]
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return seq


class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.data = [torch.tensor(text_to_sequence(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids': self.data[idx], 'label': self.labels[idx]}


# --- 3. 评价指标计算 ---
def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    # 计算 Specificity (TN / (TN + FP))
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {"Acc": acc, "Prec": prec, "Rec": rec, "Spec": spec}


# --- 4. 模型架构 ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, out_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # 拼接双向最后一步的状态
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(cat)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_filters, filter_sizes, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, emb_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)  # [batch, 1, seq_len, emb_dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


# --- 5. 训练与测试流程 ---
def train_traditional_models(X_train, X_test, y_train, y_test):
    print("\n--- 训练传统模型 (MNB, SVM, LR, RF) ---")
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "MNB": MultinomialNB(),
        "SVM": LinearSVC(class_weight='balanced', max_iter=2000),
        "LR": LogisticRegression(class_weight='balanced'),
        "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        results[name] = get_metrics(y_test, preds)
        print(f"✅ {name} 完成: Acc={results[name]['Acc']:.4f}")
    return results


def train_dl_model(model, train_loader, test_loader, model_name):
    print(f"\n--- 训练深度学习模型: {model_name} ---")
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(ids)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # 测试
    model.eval()
    y_pred = []
    y_test_list = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            outputs = model(ids)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().tolist())
            y_test_list.extend(lbl.cpu().tolist())

    metrics = get_metrics(y_test_list, y_pred)
    print(f"✅ {model_name} 完成: Acc={metrics['Acc']:.4f}")
    return metrics


# --- 6. 主程序 ---
if __name__ == "__main__":
    print(f"🚀 当前使用设备: {DEVICE}")

    # 加载数据
    texts, labels = load_spam_data(DATA_FILE_PATH)
    if texts is None: exit()

    # 划分数据集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 构建词汇表
    build_vocab(X_train)

    # 1. 传统模型
    results = train_traditional_models(X_train, X_test, y_train, y_test)

    # 准备深度学习数据
    train_ds = SpamDataset(X_train, y_train)
    test_ds = SpamDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 2. LSTM
    lstm = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LABELS)
    results["LSTM"] = train_dl_model(lstm, train_loader, test_loader, "LSTM")

    # 3. CNN
    cnn = CNNClassifier(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZES, NUM_LABELS)
    results["CNN"] = train_dl_model(cnn, train_loader, test_loader, "CNN")

    # --- 输出汇总 ---
    print("\n" + "=" * 50)
    print(f"{'Model':<10} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'Spec':<8}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:<10} | {m['Acc']:.4f}   | {m['Prec']:.4f}   | {m['Rec']:.4f}   | {m['Spec']:.4f}")