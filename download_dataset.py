"""一次性下载 opus_books 数据集并存为本地 HF 数据集目录。

运行一次：python download_dataset.py
之后提交到 git，换机器 clone 后即可离线训练，不再访问 HuggingFace。
"""
from datasets import load_dataset
from config import get_config

config = get_config()

ds_raw = load_dataset(
    f"{config['datasource']}",
    f"{config['lang_src']}-{config['lang_tgt']}",
    split='train',
)

save_dir = f"{config['datasource']}_{config['lang_src']}-{config['lang_tgt']}.hf"
ds_raw.save_to_disk(save_dir)
print(f"数据集已保存到: {save_dir}，共 {len(ds_raw)} 条样本")
