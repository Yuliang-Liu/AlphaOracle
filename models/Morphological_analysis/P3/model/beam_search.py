class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            # 可更新的情况：数量未饱和或超过最差得分
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 数量饱和需要删掉一个最差的
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成。
        best_sum_logprobs是新的候选序列中的最高得分。
        """

        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret