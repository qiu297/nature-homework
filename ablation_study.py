import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg  # 引入词性标注模块
import copy
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# --- 1. 基础配置 ---
IF_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IF_CUDA else "cpu")
MAX_LEN = 256
FIXED_BUDGET = 0.20  # 固定预算，控制变量
ABLATION_MODES = ['noun', 'verb', 'mixed']  # 三种消融模式


# --- 2. 词性攻击器类 (POS-Aware Attacker) ---
class POSAttacker:
    def __init__(self, model, vocab, cilin_path='cilin.txt'):
        self.model = model
        self.vocab = vocab
        self.syn_dict = {}
        # 模拟词林加载
        if os.path.exists(cilin_path):
            with open(cilin_path, 'r', encoding='utf-8') as f:
                for line in f:
                    p = line.strip().split(' ')
                    if len(p) > 2 and p[0][-1] == '=':
                        for w in p[1:]: self.syn_dict[w] = [x for x in p[1:] if x != w]

    def get_prob(self, text):
        # 针对LSTM模型的预测逻辑
        tokens = [self.vocab.get(w, 0) for w in jieba.lcut(text)]
        tokens = (tokens[:MAX_LEN] + [0] * MAX_LEN)[:MAX_LEN]
        input_tensor = torch.tensor([tokens]).to(DEVICE)
        with torch.no_grad():
            out = self.model(input_tensor)
            return torch.softmax(out, dim=1).cpu().numpy()[0]

    def _check_pos(self, word, mode):
        """
        核心辅助函数：检查词性是否符合当前消融模式
        """
        # 使用 jieba.posseg 获取单个词的词性
        # 注意：这里为了效率通常是对整句分词，为了演示清晰写成单词检查
        flags = [flag for _, flag in pseg.cut(word)]
        if not flags: return False
        flag = flags[0]

        if mode == 'noun':
            return flag.startswith('n')  # n, nr, ns, nt...
        elif mode == 'verb':
            return flag.startswith('v')  # v, vn, vd...
        elif mode == 'mixed':
            return True  # 混合模式不限制
        return False

    def attack(self, text, label, mode='mixed'):
        """
        params:
            mode: 'noun' (仅名词), 'verb' (仅动词), 'mixed' (混合)
        """
        # 使用 pseg 进行分词和词性标注
        words_flags = list(pseg.cut(text))
        words = [w for w, f in words_flags]
        flags = [f for w, f in words_flags]

        current_words = copy.deepcopy(words)
        orig_prob = self.get_prob(text)[label]

        max_changes = max(1, int(len(words) * FIXED_BUDGET))

        # 1. 筛选符合词性要求的候选词
        candidates_idx = []
        for i, (w, flag) in enumerate(zip(words, flags)):
            # 关键判定：根据 mode 决定是否允许修改该词
            is_target_pos = False
            if mode == 'noun' and flag.startswith('n'):
                is_target_pos = True
            elif mode == 'verb' and flag.startswith('v'):
                is_target_pos = True
            elif mode == 'mixed':
                is_target_pos = True

            if is_target_pos and w in self.syn_dict:
                candidates_idx.append(i)

        # 2. 计算重要性 (只计算筛选出的词)
        importance = []
        for i in candidates_idx:
            tmp = words[:i] + words[i + 1:]
            importance.append((i, orig_prob - self.get_prob("".join(tmp))[label]))

        importance.sort(key=lambda x: x[1], reverse=True)

        # 3. 替换逻辑
        count = 0
        changed_log = []
        for idx, _ in importance:
            if count >= max_changes: break

            old_w = current_words[idx]
            best_syn = old_w
            min_p = 1.0

            for syn in self.syn_dict[old_w]:
                current_words[idx] = syn
                new_prob = self.get_prob("".join(current_words))

                # 攻击成功判定
                if np.argmax(new_prob) != label:
                    return "".join(current_words), True, mode

                if new_prob[label] < min_p:
                    min_p = new_prob[label]
                    best_syn = syn

            if best_syn != old_w:
                current_words[idx] = best_syn
                count += 1
                changed_log.append(f"{old_w}({flags[idx]})->{best_syn}")

        return "".join(current_words), False, mode


# --- 3. 模型定义 (需与LSTM一致) ---
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


# --- 4. 消融实验主程序 ---
def run_ablation_study():
    print("\n" + "=" * 60)
    print(f"🧪 启动 4.4.3 攻击策略消融实验 (Target Model: LSTM)")
    print(f"🎯 攻击预算: {FIXED_BUDGET * 100}% | 对比模式: {ABLATION_MODES}")
    print("=" * 60)

    # 数据与词表构建
    df = pd.read_csv("test_data.csv").sample(100, random_state=42)  # 仅用测试集演示
    df['label'] = df['is_fraud'].apply(lambda x: 1 if str(x).lower() in ['1', 'true'] else 0)

    # 模拟词表 (实际应从 train_data 构建)
    all_tokens = []
    for t in df['specific_dialogue_content']: all_tokens.extend(jieba.lcut(t))
    vocab = {w: i + 1 for i, (w, _) in enumerate(Counter(all_tokens).most_common(10000))}

    # 初始化模型 (仅以此为例，实际需 load_state_dict)
    model = LSTMClassifier(len(vocab) + 2).to(DEVICE)
    model.eval()

    attacker = POSAttacker(model, vocab)

    final_results = []

    # 遍历三种模式：仅名词、仅动词、混合
    for mode in ABLATION_MODES:
        print(f"\n⚙️  正在执行模式: [{mode.upper()}] ...")
        y_true, y_att = [], []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            txt, lab = row['specific_dialogue_content'], row['label']
            p_o = np.argmax(attacker.get_prob(txt))
            y_true.append(lab)

            if p_o == lab and lab == 1:
                # 传入 mode 参数进行定向攻击
                adv, success, _ = attacker.attack(txt, lab, mode=mode)
                y_att.append(np.argmax(attacker.get_prob(adv)))
            else:
                y_att.append(p_o)

        # 统计当前模式下的指标
        metrics = {
            "Mode": mode,
            "Accuracy": accuracy_score(y_true, y_att),
            "Precision": precision_score(y_true, y_att, zero_division=0),
            "Recall": recall_score(y_true, y_att, zero_division=0)
        }

        # 计算 Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_att).ravel()
        metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        final_results.append(metrics)

    # 输出最终对比表格 (对应文档 Table 9)
    print("\n" + "=" * 60)
    print("📋 消融实验最终结果汇总")
    print("=" * 60)
    res_df = pd.DataFrame(final_results)
    # 调整列顺序
    cols = ["Mode", "Accuracy", "Precision", "Recall", "Specificity"]
    print(res_df[cols].round(4).to_string(index=False))
    res_df.to_csv("ablation_study_results.csv", index=False)


if __name__ == "__main__":
    run_ablation_study()