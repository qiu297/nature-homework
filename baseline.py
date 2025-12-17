import pandas as pd
import numpy as np
import jieba
import torch
import os
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
from collections import Counter

# --- 1. 配置和路径 ---
TRAIN_FILE_PATH = "train_data.csv"
TEST_FILE_PATH = "test_data.csv"
MAX_LEN = 256
BATCH_SIZE = 32

# 深度学习模型优化配置
LSTM_EPOCHS = 10
CNN_EPOCHS = 10
DL_LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100
NUM_LABELS = 2

DEVICE = None
jieba.setLogLevel(jieba.logging.INFO)

# 全局词汇表
VOCAB_SIZE = 0
WORD_TO_IX = {}


# --- 2. 数据加载和预处理 ---
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：文件未找到。请确保 {file_path} 在同级目录下。")
        return None, None

    df = df[['specific_dialogue_content', 'is_fraud']].copy()
    df.dropna(subset=['is_fraud'], inplace=True)
    df['specific_dialogue_content'] = df['specific_dialogue_content'].astype(str).fillna('')
    df['label'] = df['is_fraud'].astype(bool).astype(int)

    return df['specific_dialogue_content'].tolist(), df['label'].tolist()


def build_vocab(texts):
    global VOCAB_SIZE, WORD_TO_IX
    all_words = []
    for text in texts:
        all_words.extend(list(jieba.cut(text)))
    word_counts = Counter(all_words)
    vocab = ['<pad>', '<unk>'] + [word for word, count in word_counts.items() if count >= 2]
    WORD_TO_IX = {word: i for i, word in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)
    print(f"词汇表构建完成，大小: {VOCAB_SIZE}")


def get_metrics(y_true, y_pred):
    """计算四个核心指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)

    # 计算 Specificity (特异度)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        specificity = 0.0

    return {
        'Accuracy': acc,
        'Precision': p,
        'Recall': r,
        'Specificity': specificity
    }


# --- 3. 传统模型训练与评估 ---
def train_and_test_traditional_models(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)),
                                 token_pattern=None, min_df=3)

    print("\n[Traditional Models] 正在生成 TF-IDF 特征...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}
    models_to_run = [
        ('MNB', MultinomialNB()),
        ('SVM', LinearSVC(random_state=42, dual=False)),
        ('LR', LogisticRegression(random_state=42, max_iter=1000)),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    for name, model in models_to_run:
        print(f"正在训练 {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        results[name] = get_metrics(y_test, y_pred)

    return results


# --- 4. 深度学习模型定义 ---
class DialogueDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        token_indices = [WORD_TO_IX.get(word, WORD_TO_IX['<unk>'])
                         for word in list(jieba.cut(self.texts[item]))]
        if len(token_indices) > self.max_len:
            token_indices = token_indices[:self.max_len]
        else:
            token_indices += [WORD_TO_IX['<pad>']] * (self.max_len - len(token_indices))
        return {'input_ids': torch.tensor(token_indices, dtype=torch.long),
                'labels': torch.tensor(self.labels[item], dtype=torch.long)}


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        hidden_combined = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden_combined)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        return self.fc(cat)


# --- 5. 训练与保存逻辑 ---
def train_deep_model(model, X_train, y_train, X_test, y_test, model_name, epochs):
    train_dataset = DialogueDataset(X_train, y_train, MAX_LEN)
    test_dataset = DialogueDataset(X_test, y_test, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[{model_name}] 开始训练...")
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            ids, labels = batch['input_ids'].to(DEVICE), batch['labels'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(DEVICE)
            outputs = model(ids)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().tolist())

    metrics = get_metrics(y_test, y_pred)

    # 保存权重
    save_path = f"{model_name.lower()}_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ {model_name} 权重已保存至 {save_path}")

    return metrics


# --- 6. 主程序 ---
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 当前使用设备: {DEVICE}")

    X_train, y_train = load_and_preprocess_data(TRAIN_FILE_PATH)
    X_test, y_test = load_and_preprocess_data(TEST_FILE_PATH)
    if X_train is None: exit()

    build_vocab(X_train)
    results = {}

    # 1-4. 传统模型
    results.update(train_and_test_traditional_models(X_train, X_test, y_train, y_test))

    # 5. LSTM
    lstm = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LABELS)
    results['LSTM'] = train_deep_model(lstm, X_train, y_train, X_test, y_test, 'LSTM', LSTM_EPOCHS)

    # 6. CNN
    cnn = CNNClassifier(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZES, NUM_LABELS)
    results['CNN'] = train_deep_model(cnn, X_train, y_train, X_test, y_test, 'CNN', CNN_EPOCHS)

    # 7. 结果展示
    print("\n" + "=" * 70)
    print("                6大基线模型评估结果汇总（含四个指标）")
    print("=" * 70)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))