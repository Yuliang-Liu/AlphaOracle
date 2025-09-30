import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout, normalize_before, pad_token_id, num_classes, max_position_embeddings, 
                 return_intermediate_dec, eos_index, activation="relu",beam_search=False,beam_size=100,beam_search_max_length=24,use_length_penalty=False,length_penalty=0.7):
        super(Transformer, self).__init__()
        self.embedding = DecoderEmbeddings(num_classes, d_model, pad_token_id, max_position_embeddings, dropout)
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.nhead = nhead
        self.d_model = d_model
        self.eos_index = eos_index
        self.pad_token_id = pad_token_id
        self.num_encoder_layers = num_encoder_layers
        self.max_position_embeddings = max_position_embeddings
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.beam_search_max_length = beam_search_max_length
        self.use_length_penalty = use_length_penalty
        self.length_penalty = length_penalty

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, seq, vocab_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed.half())
        else:
            memory = src

        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)
        if self.training:
            tgt = self.embedding(seq).permute(1, 0, 2)
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed[:len(tgt)],
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
            # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
            return vocab_embed(hs[-1].transpose(0, 1))
        else:
            if self.beam_search==False:
                probs = []
                for i in range(self.max_position_embeddings):
                    tgt = self.embedding(seq).permute(1, 0, 2)
                    hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                            pos=pos_embed, query_pos=query_embed[:len(tgt)],
                            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
                    out = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
                    out = out.softmax(-1)

                    prob, extra_seq = out.topk(dim=-1, k=1)
                    seq = torch.cat([seq, extra_seq], dim=-1)
                    probs.append(prob)                
                    if extra_seq[0] == self.eos_index or i>24:
                        break
                
                seq = seq[:, 1:] # remove start index
                return seq, torch.cat(probs, dim=-1)
            else:
                num_beams= self.beam_size
                max_length = self.beam_search_max_length
                length_penalty = self.length_penalty
                use_length_penalty = self.use_length_penalty
                batch_size=1
                generated_hyps = [
                    BeamHypotheses(num_beams, max_length, length_penalty, use_length_penalty)
                    for _ in range(batch_size)
                ]
                # 每个beam容器的得分，共batch_size*num_beams个
                beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=seq.device)
                beam_scores = beam_scores.view(-1)

                # 每个样本是否完成生成，共batch_size个
                done = [False for _ in range(batch_size)]
                # input_ids = seq.expand(num_beams, -1)  # [beam_size, seq_len]
                input_ids=seq
                for cur_len in range(1,self.beam_search_max_length+1):
                    if cur_len==1:
                        tgt = self.embedding(seq).permute(1, 0, 2)
                        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                pos=pos_embed, query_pos=query_embed[:len(tgt)],
                                tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
                        logits = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
                        log_probs = F.log_softmax(logits, dim=-1)  # 用log概率便于累加 # [beam_size, vocab_size]
                        vocab_size=log_probs.size(-1)
                        next_scores = log_probs
                        next_scores = next_scores.view(
                            batch_size,  vocab_size
                        )  # (batch_size, vocab_size)
                    else:
                        tgt = self.embedding(input_ids).permute(1, 0, 2)  # [seq_len, beam_size, embed_dim]
                        hs = self.decoder(
                            tgt, memory.repeat(1, num_beams, 1),  # 扩展memory以匹配beam_size
                            memory_key_padding_mask=mask.repeat(num_beams, 1),
                            pos=pos_embed.repeat(1, num_beams, 1),
                            query_pos=query_embed[:len(tgt)].repeat(1, num_beams, 1),
                            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device)
                        )  # [seq_len, beam_size, embed_dim]
                        logits = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
                        log_probs = F.log_softmax(logits, dim=-1)  # 用log概率便于累加 # [beam_size, vocab_size]
                        # logits = vocab_embed(hs[-1].transpose(0, 1))  # 取最后一个时间步
                        # log_probs = torch.log_softmax(logits, dim=-1)  
                        vocab_size=log_probs.size(-1)
                        next_scores = log_probs + beam_scores[:, None].expand_as(log_probs) 
                        # 为了提速，将结果重排成图中3的形状
                        next_scores = next_scores.view(
                                batch_size, num_beams * vocab_size
                            )  # (batch_size, num_beams * vocab_size)
                    # 取出分数最高的token（图中黑点）和其对应得分
                    # sorted=True，保证返回序列是有序的
                    next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=
                                                        True, sorted=True)
                    # 下一个时间步整个batch的beam列表
                    # 列表中的每一个元素都是三元组
                    # (分数, token_id, beam_id)
                    next_batch_beam = []

                    # 对每一个样本进行扩展
                    for batch_idx in range(batch_size):

                        # 检查样本是否已经生成结束
                        if done[batch_idx]:
                            # 对于已经结束的句子，待添加的是pad token
                            next_batch_beam.extend([(0, self.pad_token_id, 0)] * num_beams)  # pad the batch
                            continue

                        # 当前样本下一个时间步的beam列表
                        next_sent_beam = []

                        # 对于还未结束的样本需要找到分数最高的num_beams个扩展
                        # 注意，next_scores和next_tokens是对应的
                        # 而且已经按照next_scores排好顺序
                        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                            zip(next_tokens[batch_idx], next_scores[batch_idx])
                        ):
                            # get beam and word IDs
                            # 这两行可参考图中3进行理解
                            beam_id = beam_token_id // vocab_size 
                            token_id = beam_token_id % vocab_size

                            effective_beam_id = batch_idx * num_beams + beam_id

                            # 如果出现了EOS token说明已经生成了完整句子
                            if (self.eos_index is not None) and (token_id.item() == self.eos_index):
                                # if beam_token does not belong to top num_beams tokens, it should not be added
                                is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                                if is_beam_token_worse_than_top_num_beams:
                                    continue
                                # 往容器中添加这个序列
                                generated_hyps[batch_idx].add(
                                    input_ids[effective_beam_id].clone(), beam_token_score.item(),
                                )
                            else:
                                # add next predicted word if it is not eos_token
                                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                            # 扩展num_beams个就够了
                            if len(next_sent_beam) == num_beams:
                                break

                        # 检查这个样本是否已经生成完了，有两种情况
                        # 1. 已经记录过该样本结束
                        # 2. 新的结果没有使结果改善
                        done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                            next_scores[batch_idx].max().item(), cur_len=cur_len
                        )

                        # 把当前样本的结果添加到batch结果的后面
                        next_batch_beam.extend(next_sent_beam)

                    # 如果全部样本都已经生成结束便可以直接退出了
                    if all(done):
                        break
                    
                    # 把三元组列表再还原成三个独立列表
                    beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
                    beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
                    beam_idx = input_ids.new([x[2] for x in next_batch_beam])

                    # 准备下一时刻的解码器输入
                    # 取出实际被扩展的beam
                    input_ids = input_ids[beam_idx, :]
                    # 在这些beam后面接上新生成的token
                    input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
                # select the best hypotheses，最终输出
                # 每个样本返回几个句子
                output_num_return_sequences_per_batch = 100
                # 记录每个返回句子的长度，用于后面pad
                output_batch_size=output_num_return_sequences_per_batch*batch_size
                sent_lengths = input_ids.new(output_batch_size)
                best_sequences = []
                best_scores = []
                # 对每个样本取出最好的output_num_return_sequences_per_batch个句子
                for i, hypotheses in enumerate(generated_hyps):
                    sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
                    for j in range(output_num_return_sequences_per_batch):
                        effective_batch_idx = output_num_return_sequences_per_batch * i + j
                        # best_hyp = sorted_hyps.pop()[1]
                        score,best_hyp = sorted_hyps.pop()
                        sent_lengths[effective_batch_idx] = len(best_hyp)
                        best_sequences.append(best_hyp)
                        best_scores.append(score)

                # 如果长短不一则pad句子，使得最后返回结果的长度一样
                if sent_lengths.min().item() != sent_lengths.max().item():
                    sent_max_len = min(sent_lengths.max().item() + 1, max_length)
                    # 先把输出矩阵填满PAD token
                    decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.pad_token_id)

                    # 填入真正的内容
                    for i, hypo in enumerate(best_sequences):
                        decoded[i, : sent_lengths[i]] = hypo
                        # 填上eos token
                        if sent_lengths[i] < max_length:
                            decoded[i, sent_lengths[i]] = self.eos_index
                else:
                    # 所有生成序列都还没结束，直接堆叠即可
                    decoded = torch.stack(best_sequences).type(torch.long).to(next(self.parameters()).device)
                # 将分数转换为概率（从对数概率转换回来）
                best_probs = torch.tensor(best_scores, device=decoded.device).exp()
                # 返回的结果包含BOS token
                return decoded, best_probs


            # else:
            #     # 改为beam search实现
            #     beam_size = 100
            #     # 初始化beam (序列, 累计概率)
            #     beams = [(seq, 0.0)]  # seq初始为[start_token]
            #     final_results = []
                
            #     for step in range(self.max_position_embeddings):
            #         new_beams = []
            #         for seq, score in beams:
            #             if seq[0, -1] == self.eos_index:  # 已结束的序列
            #                 final_results.append((seq, score))
            #                 if len(final_results) >= 100:
            #                     break
            #                 continue
                            
            #             tgt = self.embedding(seq).permute(1, 0, 2)
            #             hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
            #                         pos=pos_embed, query_pos=query_embed[:len(tgt)],
            #                         tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
            #             logits = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
            #             probs = F.log_softmax(logits, dim=-1)  # 用log概率便于累加
                        
            #             # 取topk候选
            #             topk_probs, topk_indices = probs.topk(k=beam_size, dim=-1)
            #             for i in range(beam_size):
            #                 new_seq = torch.cat([seq, topk_indices[:, i:i+1]], dim=-1)
            #                 new_score = score + topk_probs[:, i].item()
            #                 new_beams.append((new_seq, new_score))
                    
            #         # 保留总概率最高的beam_size个路径
            #         new_beams.sort(key=lambda x: -x[1])
            #         beams = new_beams[:beam_size]
                    
            #         if all(beam[0][0, -1] == self.eos_index for beam in beams):
            #             break
                
            #     # 合并最终结果
            #     final_results.extend(beams)
            #     # 只保留已结束的序列
            #     final_results = [r for r in final_results if r[0][0, -1] == self.eos_index]
            #     final_results.sort(key=lambda x: -x[1])
            #     final_results = final_results[:100]
            #     output_seqs = [res[0][0, 1:] for res in final_results]  # 去掉<start>
            #     output_scores = [res[1] for res in final_results]       # 累计log概率

            #     # 转换为概率（可选：从log概率转回概率）
            #     output_probs = torch.exp(torch.tensor(output_scores))  # 用exp恢复概率值
            #     return output_seqs, output_probs
            # else:
            #     beam_size = 100
            #     beams = [(seq, 0.0)]*beam_size  # (序列, 累计log概率)
            #     final_results = []

            #     for step in range(self.max_position_embeddings):
            #         if len(final_results) >= beam_size:
            #             break

            #         # 将当前所有候选序列拼接成 Batch (shape: [batch_size, seq_len])
            #         batch_seq = torch.cat([beam[0] for beam in beams], dim=0)  # [beam_size, seq_len]
            #         batch_score = torch.tensor([beam[1] for beam in beams])     # [beam_size]

            #         # 批量计算 decoder (一次处理所有候选序列)
            #         tgt = self.embedding(batch_seq).permute(1, 0, 2)  # [seq_len, beam_size, emb_dim]
            #         hs = self.decoder(
            #                     tgt, memory.repeat(1, beam_size, 1),  # 扩展memory以匹配beam_size
            #                     memory_key_padding_mask=mask.repeat(beam_size, 1),
            #                     pos=pos_embed.repeat(1, beam_size, 1),
            #                     query_pos=query_embed[:len(tgt)].repeat(1, beam_size, 1),
            #                     tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device)
            #                 )  # [seq_len, beam_size, embed_dim]
                    
            #         # 计算下一个词的概率 (shape: [beam_size, vocab_size])
            #         logits = vocab_embed(hs[-1].transpose(0, 1))  # 取最后一个时间步
            #         log_probs = torch.log_softmax(logits, dim=-1)  # [beam_size, vocab_size]

            #         # 并行计算 TopK (全局 TopK，而不是逐序列 TopK)
            #         # 计算所有可能的扩展路径 (beam_size * vocab_size 种可能)
            #         expanded_scores = (batch_score.unsqueeze(1) + log_probs).view(-1)  # [beam_size * vocab_size]
            #         topk_scores, topk_indices = torch.topk(expanded_scores, k=beam_size)  # 全局 Top100

            #         # 解码 TopK 对应的 (beam_idx, token_idx)
            #         beam_indices = topk_indices // log_probs.size(1)  # 属于哪个 beam
            #         token_indices = topk_indices % log_probs.size(1)   # 对应的 token

            #         # 生成新序列
            #         new_beams = []
            #         for i in range(beam_size):
            #             beam_idx = beam_indices[i].item()
            #             token_idx = token_indices[i].item()
            #             new_seq = torch.cat([batch_seq[beam_idx], token_idx.unsqueeze(0).unsqueeze(0)], dim=-1)
            #             new_score = topk_scores[i].item()
            #             new_beams.append((new_seq, new_score))

            #         # 更新 beams
            #         beams = new_beams

            #         # 提前终止条件：所有候选都已结束
            #         if all(beam[0][0, -1] == self.eos_index for beam in beams):
            #             break

            #     # 合并最终结果
            #     final_results.extend([r for r in beams if r[0][0, -1] == self.eos_index])
            #     final_results.sort(key=lambda x: -x[1])
            #     final_results = final_results[:beam_size]

            #     best_seq = final_results[0][0][:, 1:] if final_results else beams[0][0][:, 1:]
            #     return best_seq, None



class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    max_position_embeddings = (2 + 25) * args.max_num_text_ins + 1
    return Transformer(
        d_model=args.tfm_hidden_dim,
        nhead=args.tfm_nheads,
        num_encoder_layers=args.tfm_enc_layers,
        num_decoder_layers=args.tfm_dec_layers,
        dim_feedforward=args.tfm_dim_feedforward,
        dropout=args.tfm_dropout,
        normalize_before=args.tfm_pre_norm,
        pad_token_id=args.padding_index,
        num_classes=args.num_classes,
        max_position_embeddings=max_position_embeddings,
        return_intermediate_dec=False,
        eos_index=args.eos_index,
        beam_search=args.beam_search,
        beam_size=args.beam_size,
        beam_search_max_length=args.beam_search_max_length,
        use_length_penalty=args.use_length_penalty,
        length_penalty=args.length_penalty,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, use_length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
        self.length_penalty= length_penalty
        self.use_length_penalty = use_length_penalty

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        if self.use_length_penalty:
            score = sum_logprobs / len(hyp) ** self.length_penalty
        else:
            score = sum_logprobs
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
            if self.use_length_penalty:
                cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            else:
                cur_score = best_sum_logprobs
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret