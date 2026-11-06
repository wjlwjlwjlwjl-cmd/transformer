import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # 将特殊符号转换为 token id
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) # 开始标志
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64) # 结束标志
        self.pad_token_src = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64) # 训练词库的 [POD] 对应的 token id
        self.pad_token_tgt = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64) # 没达到 set_len 的位置使用 [PAD] 占位

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # 将文本转换为 token 并拿到 id
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 编码文本头尾添加 [SOS]、[EOS]
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 
        # 译码只需要 [SOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # 当切分出的 token 数量超过 seq_len 时，会出现 padding_tokens 为负数的情况
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # [[SOS], X, Y, [EOS], [PAD]]
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_src] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # [[SOS], X, Y, [PAD]]
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # [X, Y, [EOS], [PAD]]，Decoder 输出的预期正确答案
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token_src).bool().unsqueeze(0).unsqueeze(0), # 后续编码掩码与注意力（Shape batch_size, h, seq_len, seq_len）做矩阵乘法，这里给予广播机制完成
            "decoder_mask": (decoder_input != self.pad_token_tgt).bool().unsqueeze(0) & causal_mask(decoder_input.size(0)), # 加上
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.bool) # diagonal 表示取主对角线向上1个单位的所有元素
    return ~mask # 为 0 的位置，是当前位置及之前的位置，允许关注