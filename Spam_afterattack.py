import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import copy
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import nltk
from nltk.corpus import wordnet

# 首次运行建议下载 wordnet 数据
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# --- 1. 配置与硬件检测 ---
IF_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IF_CUDA else "cpu")
MAX_LEN = 150
ATTACK_BUDGET = 0.20  # 20% 修改预算


def clean_text(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)  # 返回单词列表


# --- 2. 深度学习模型结构 ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(cat)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.convs = nn.ModuleList([nn.Conv2d(1, 100, (fs, 100)) for fs in [3, 4, 5]])
        self.fc = nn.Linear(3 * 100, 2)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        return self.fc(torch.cat(x, 1))


# --- 3. 对抗攻击核心类 ---
class SpamAttacker:
    def __init__(self, model, model_type, vectorizer=None, word_to_idx=None):
        self.model = model
        self.model_type = model_type  # 'trad' (传统) 或 'dl' (深度学习)
        self.vectorizer = vectorizer
        self.word_to_idx = word_to_idx

    def get_prob(self, text_list):
        """获取模型对某段文本（词列表）的预测概率"""
        text_str = " ".join(text_list)
        if self.model_type == 'trad':
            vec = self.vectorizer.transform([text_str])
            # LinearSVC 没有 predict_proba，使用 decision_function 模拟
            if isinstance(self.model, LinearSVC):
                dec = self.model.decision_function(vec)[0]
                prob = 1 / (1 + np.exp(-dec))  # Sigmoid
                return np.array([1 - prob, prob])
            return self.model.predict_proba(vec)[0]
        else:
            self.model.eval()
            ids = [self.word_to_idx.get(w, 1) for w in text_list]
            if len(ids) < MAX_LEN:
                ids += [0] * (MAX_LEN - len(ids))
            else:
                ids = ids[:MAX_LEN]
            input_tensor = torch.tensor([ids]).to(DEVICE)
            with torch.no_grad():
                out = self.model(input_tensor)
                return torch.softmax(out, dim=1).cpu().numpy()[0]

    def get_synonyms(self, word):
        """基于 WordNet 获取英文同义词"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                name = l.name().replace('_', ' ')
                if name.lower() != word.lower():
                    synonyms.add(name)
        return list(synonyms)

    def attack(self, text_list, target_label):
        """攻击逻辑：寻找重要词并替换"""
        origin_prob = self.get_prob(text_list)
        current_text = copy.deepcopy(text_list)

        # 1. 计算重要性 (Leave-one-out)
        importance = []
        for i, word in enumerate(text_list):
            leave_one_text = text_list[:i] + text_list[i + 1:]
            prob_after = self.get_prob(leave_one_text)
            importance.append((i, origin_prob[target_label] - prob_after[target_label]))

        # 按重要性降序排序
        importance.sort(key=lambda x: x[1], reverse=True)

        # 2. 尝试替换 (受预算限制)
        max_changes = max(1, int(len(text_list) * ATTACK_BUDGET))
        change_count = 0
        changed_words = []

        for idx, score in importance:
            if change_count >= max_changes: break

            orig_word = current_text[idx]
            syns = self.get_synonyms(orig_word)

            best_syn = None
            min_target_prob = 1.0

            for s in syns:
                temp_text = copy.deepcopy(current_text)
                temp_text[idx] = s
                new_prob = self.get_prob(temp_text)
                if np.argmax(new_prob) != target_label:
                    current_text[idx] = s
                    return current_text, True, [f"{orig_word}->{s}"]

                if new_prob[target_label] < min_target_prob:
                    min_target_prob = new_prob[target_label]
                    best_syn = s

            if best_syn:
                current_text[idx] = best_syn
                changed_words.append(f"{orig_word}->{best_syn}")
                change_count += 1

        return current_text, np.argmax(self.get_prob(current_text)) != target_label, changed_words


# --- 4. 实验主逻辑 ---
def run_spam_experiment():
    # A. 数据加载
    df = pd.read_csv("spam.csv", encoding='latin-1').iloc[:, :2]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 这里仅选取测试集部分进行攻击演示
    test_df = df.sample(n=500, random_state=42)

    # B. 传统模型准备 (假设已经训练好，这里简略初始化)
    vectorizer = TfidfVectorizer(max_features=3000).fit(df['text'])
    X_tfidf = vectorizer.transform(df['text'])
    y = df['label'].values

    trad_models = {
        "MNB": MultinomialNB().fit(X_tfidf, y),
        "SVM": LinearSVC(class_weight='balanced').fit(X_tfidf, y),
        "LR": LogisticRegression().fit(X_tfidf, y),
        "RF": RandomForestClassifier(n_estimators=50).fit(X_tfidf, y)
    }

    # C. 深度学习模型准备 (需要词汇表)
    all_words = []
    for t in df['text']: all_words.extend(clean_text(t))
    counts = Counter(all_words)
    word_to_idx = {w: i + 2 for i, (w, c) in enumerate(counts.most_common(10000))}
    word_to_idx["<PAD>"], word_to_idx["<UNK>"] = 0, 1

    dl_models = {
        "LSTM": LSTMClassifier(len(word_to_idx)).to(DEVICE),
        "CNN": CNNClassifier(len(word_to_idx)).to(DEVICE)
    }
    # 注意：实际使用需用 torch.load 加载您训练好的 .pth 文件

    # D. 执行攻击循环
    all_results = []

    for name, model in {**trad_models, **dl_models}.items():
        print(f"🕵️ 正在评估模型攻击韧性: {name}")
        m_type = 'trad' if name in trad_models else 'dl'
        attacker = SpamAttacker(model, m_type, vectorizer, word_to_idx)

        y_true, y_att = [], []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            txt_list = clean_text(row['text'])
            lab = row['label']

            p_o = np.argmax(attacker.get_prob(txt_list))
            y_true.append(lab)

            # 只针对识别正确的 Spam 邮件发动攻击
            if p_o == lab and lab == 1:
                adv, success, _ = attacker.attack(txt_list, lab)
                y_att.append(np.argmax(attacker.get_prob(adv)))
            else:
                y_att.append(p_o)

        # 计算指标
        acc = accuracy_score(y_true, y_att)
        rec = recall_score(y_true, y_att)
        pre = precision_score(y_true, y_att, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_att).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        all_results.append({"Model": name, "Acc": acc, "Prec": pre, "Rec": rec, "Spec": spec})

    # E. 打印结果
    res_df = pd.DataFrame(all_results)
    print("\n📊 对抗攻击后的 Spam 数据集性能总结:")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_spam_experiment()