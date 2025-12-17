import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jieba
import copy
from collections import Counter
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# --- 1. 配置与硬件检测 ---
IF_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IF_CUDA else "cpu")
VOCAB_SIZE, MAX_LEN = 15000, 256
ATTACK_BUDGET = 0.20  # 20% 修改预算


def print_device_info():
    print("=" * 65)
    print(f"💻 硬件报告: {'✅ GPU 加速模式' if IF_CUDA else '🐢 CPU 模式'}")
    if IF_CUDA: print(f"🚀 显卡型号: {torch.cuda.get_device_name(0)}")
    print("=" * 65)


# --- 2. 模型结构 ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h, _) = self.lstm(embedded)
        return self.fc(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(128, 100, fs) for fs in [3, 4, 5]])
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        e = self.embedding(x).permute(0, 2, 1)
        return self.fc(torch.cat([torch.max(torch.relu(conv(e)), dim=2)[0] for conv in self.convs], dim=1))


# --- 3. 诊断增强型攻击类 ---
class SynonymAttacker:
    def __init__(self, model, model_type, vocab, tfidf, cilin_path='cilin.txt'):
        self.model = model
        self.model_type = model_type
        self.vocab = vocab
        self.tfidf = tfidf
        self.syn_dict = {}
        if os.path.exists(cilin_path):
            with open(cilin_path, 'r', encoding='utf-8') as f:
                for line in f:
                    p = line.strip().split(' ')
                    if len(p) > 2 and p[0][-1] == '=':
                        for w in p[1:]: self.syn_dict[w] = [x for x in p[1:] if x != w]

    def get_prob(self, text):
        if self.model_type == 'ml':
            return self.model.predict_proba(self.tfidf.transform([text]))[0]
        else:
            max_idx = self.model.embedding.num_embeddings
            tokens = [self.vocab.get(w, 0) for w in jieba.lcut(text)]
            tokens = [t if t < max_idx else 0 for t in tokens]
            tokens = (tokens[:MAX_LEN] + [0] * MAX_LEN)[:MAX_LEN]
            with torch.no_grad():
                out = self.model(torch.LongTensor([tokens]).to(DEVICE))
                return torch.softmax(out, dim=1).cpu().numpy()[0]

    def attack(self, text, label):
        words = jieba.lcut(text)
        current_words = copy.deepcopy(words)
        change_info = []
        importance = []
        orig_prob = self.get_prob(text)[label]
        for i, w in enumerate(words):
            if w in self.syn_dict:
                tmp = words[:i] + words[i + 1:]
                importance.append((i, orig_prob - self.get_prob("".join(tmp))[label]))
        importance.sort(key=lambda x: x[1], reverse=True)
        limit = max(1, int(len(words) * ATTACK_BUDGET))
        count = 0
        for idx, _ in importance:
            if count >= limit: break
            old_w = current_words[idx]
            best_syn, min_p = old_w, self.get_prob("".join(current_words))[label]
            for cand in self.syn_dict[old_w]:
                current_words[idx] = cand
                p = self.get_prob("".join(current_words))[label]
                if p < min_p:
                    min_p, best_syn = p, cand
                if np.argmax(self.get_prob("".join(current_words))) != label:
                    change_info.append(f"{old_w}->{cand}")
                    return "".join(current_words), True, change_info
            if best_syn != old_w:
                change_info.append(f"{old_w}->{best_syn}")
                count += 1
            current_words[idx] = best_syn
        return "".join(current_words), False, change_info


# --- 4. 实验主程序 ---
def run_evaluation():
    print_device_info()
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv").sample(100)  # 选取100个样本进行攻击评估

    train_df['label'] = train_df['is_fraud'].apply(lambda x: 1 if str(x).lower() in ['1', 'true', 'fraud'] else 0)
    test_df['label'] = test_df['is_fraud'].apply(lambda x: 1 if str(x).lower() in ['1', 'true', 'fraud'] else 0)

    tfidf = TfidfVectorizer(max_features=5000, tokenizer=jieba.lcut, token_pattern=None).fit(
        train_df['specific_dialogue_content'])
    all_tokens = []
    for t in train_df['specific_dialogue_content']: all_tokens.extend(jieba.lcut(t))
    vocab = {w: i + 1 for i, (w, _) in enumerate(Counter(all_tokens).most_common(VOCAB_SIZE - 1))}

    # 包含 RF 在内的 6 个模型配置
    models_config = [
        ("MNB", MultinomialNB(), 'ml'),
        ("SVM", SVC(kernel='linear', probability=True), 'ml'),
        ("LR", LogisticRegression(), 'ml'),
        ("RF", RandomForestClassifier(), 'ml'),
        ("LSTM", None, 'dl'),
        ("CNN", None, 'dl')
    ]

    all_results = []
    print("\n🔍 正在进行 20% 预算下的同义词改写攻击测试...")

    for name, m_obj, m_type in models_config:
        print(f"评估进度: [{name}] 处理中...")
        if m_type == 'ml':
            m_obj.fit(tfidf.transform(train_df['specific_dialogue_content']), train_df['label'])
            model = m_obj
        else:
            path = f"{name.lower()}_model.pth"
            if not os.path.exists(path):
                print(f"⚠️ 跳过 {name}: 未找到权重文件 {path}")
                continue
            ckpt = torch.load(path, map_location=DEVICE)
            model = LSTMClassifier(ckpt['embedding.weight'].shape[0]) if name == "LSTM" else CNNClassifier(
                ckpt['embedding.weight'].shape[0])
            model.load_state_dict(ckpt)
            model.to(DEVICE).eval()

        attacker = SynonymAttacker(model, m_type, vocab, tfidf)
        y_true, y_att = [], []
        examples = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {name} 攻击测试"):
            txt, lab = row['specific_dialogue_content'], row['label']
            # 获取原始预测结果
            p_o = np.argmax(attacker.get_prob(txt))
            y_true.append(lab)

            # 攻击逻辑：只针对原本识别正确的正类（欺诈）样本发起改写
            if p_o == lab and lab == 1:
                adv, success, changes = attacker.attack(txt, lab)
                p_a = np.argmax(attacker.get_prob(adv))
                y_att.append(p_a)
                if success and p_a == 0:
                    examples.append({"old": txt, "new": adv, "changes": changes})
            else:
                y_att.append(p_o)  # 其他样本保持原样

        # 诊断输出（帮助分析原因）
        if examples:
            sample = examples[0]
            print(f"   💡 攻击成功典型替换: {', '.join(sample['changes'][:3])}")

        # 计算四个指标
        acc = accuracy_score(y_true, y_att)
        pre = precision_score(y_true, y_att, zero_division=0)
        rec = recall_score(y_true, y_att, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_att).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        all_results.append({
            "Model": name,
            "Acc": acc,
            "Pre": pre,
            "Rec": rec,
            "Spec": spec
        })

    # 输出表格：仅包含 After_Attack 数据
    df_res = pd.DataFrame(all_results).round(4)
    print("\n" + "=" * 80)
    print(f"📊 同义词改写攻击实验结果 (20% 修改预算)")
    print("=" * 80)
    print(df_res)
    print("=" * 80)
    df_res.to_csv("after_attack_4metrics_results.csv", index=False)


if __name__ == "__main__":
    run_evaluation()