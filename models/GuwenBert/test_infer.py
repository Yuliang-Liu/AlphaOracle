import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 指定保存模型的路径
model_path = "path/to/model"

# 加载微调后的 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)

def predict_masked_from_candidates(sentence: str,
                                   mask_position: int,
                                   candidates: list):
    """
    在指定位置插入 [MASK]，仅从给定的候选字列表中选出概率
    :param sentence: 输入的完整句子（支持无空格中文句子，保留 [UNK]）
    :param mask_position: 要 mask 的 token 位置（从0开始计），含 [UNK]
    :param candidates: 候选字列表，例如 ['伐', '伐', '放']
    :return: 按概率排序的候选字列表 [{'Rank': 1, 'Category': '伐', 'Possibility': 0.8861}, ...]
    """
    import re
    tokens = re.split(r'(\[UNK\])', sentence.strip())
    tokens = [t for token in tokens
                 for t in (list(token) if token != '[UNK]' else [token])]

    if not (0 <= mask_position < len(tokens)):
        raise ValueError(f"mask_position 超出范围，应在 [0, {len(tokens)-1}]")

    tokens[mask_position] = tokenizer.mask_token
    masked_sentence = "".join(tokens)

    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if mask_token_index.numel() == 0:
        raise ValueError("句子中未找到 mask token")
    mask_pos = mask_token_index.item()

    probs = F.softmax(logits[0, mask_pos], dim=-1)

    candidate_ids = tokenizer.convert_tokens_to_ids(candidates)
    candidate_probs = {cand: probs[token_id].item()
                       for cand, token_id in zip(candidates, candidate_ids)}

    sorted_candidates = sorted(candidate_probs.items(), key=lambda x: x[1], reverse=True)
    results = [
        {"Rank": i + 1, "Category": cand, "Possibility": prob}
        for i, (cand, prob) in enumerate(sorted_candidates)
    ]

    return results

# example
example_sentence = "王子卜宾贞辛亥王入自[UNK]王[UNK]有[UNK]唯害一月"
mask_pos = 0
cands = ["伐", "伤", "侵", "害", "放"]

results = predict_masked_from_candidates(example_sentence, mask_pos, cands)
for item in results:
    print(item)
