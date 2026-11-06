"""英文 -> 俄文 机器翻译推理脚本

用法：
    python translate.py "It was his want of faith."
    python translate.py               # 不带参数则进入交互模式，逐行输入英文

依赖：需在有 torch/tokenizers 的 conda 环境中运行（如 transformer 环境）。
"""
import sys
import torch
from pathlib import Path

from tokenizers import Tokenizer
from transformer import build_transformer
from config import get_config, latest_weights_file_path


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(decoder_input, source_mask, decoder_mask, encoder_output)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def translate(model, tokenizer_src, tokenizer_tgt, text, device, max_len):
    model.eval()
    with torch.no_grad():
        # 编码源句（含 SOS/EOS，padding 到 seq_len）
        enc_tokens = tokenizer_src.encode(text).ids
        num_padding = max_len - len(enc_tokens) - 2
        encoder_input = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(enc_tokens, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * num_padding, dtype=torch.int64),
        ], dim=0).unsqueeze(0).to(device)  # [1, seq_len]

        encoder_mask = (encoder_input != tokenizer_src.token_to_id('[PAD]')).bool().unsqueeze(0).unsqueeze(0).to(device)

        out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
        return tokenizer_tgt.decode(out_tokens.detach().cpu().numpy())


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 tokenizer
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    # 构建模型并加载最新权重
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model'],
    ).to(device)

    weights = latest_weights_file_path(config)
    if weights is None:
        print("[错误] 未找到权重文件，请先训练或确认权重目录。")
        sys.exit(1)

    print(f"加载权重: {weights}")
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    max_len = config['seq_len']
    print(f"设备: {device}，模型就绪，输入英文开始翻译（Ctrl+C 退出）\n")

    # 命令行参数模式
    if len(sys.argv) > 1:
        src = " ".join(sys.argv[1:])
        print(f"EN: {src}")
        print(f"RU: {translate(model, tokenizer_src, tokenizer_tgt, src, device, max_len)}")
        return

    # 交互模式
    while True:
        try:
            src = input("EN > ").strip()
            if not src:
                continue
            print(f"RU: {translate(model, tokenizer_src, tokenizer_tgt, src, device, max_len)}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
