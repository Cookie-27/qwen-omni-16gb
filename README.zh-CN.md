# 在 16GB 显卡上运行 Qwen Omni 候选评分

[![CPU 验证](https://github.com/Cookie-27/qwen-omni-16gb/actions/workflows/verify.yml/badge.svg)](https://github.com/Cookie-27/qwen-omni-16gb/actions/workflows/verify.yml)

[English](README.md) · [上手与排错](docs/quickstart.md) · [测量证据](docs/evidence.md)

这是针对 Qwen2.5-Omni-3B Thinker 的小型推理示例。通过把文本注意力按头分批调用原 SDPA，降低某些回退实现的峰值显存；每个头保留完整序列和对应 KV 头，音频与视觉注意力仍使用原实现。

历史 RTX 5080 实测：完整约 40–60 秒音频、全部选定的每秒一帧图像，最长 5,521 token；PyTorch 分配峰值 10.202 GiB，整卡采样峰值 12,001 MiB。加载的 Thinker 实际为约 4.703B 参数。

**范围是单次前向、末位候选答案评分。** 不包含语音生成、任意长文本生成或训练的显存承诺。

无需 GPU 或额外依赖即可核验历史结果：

```bash
git clone https://github.com/Cookie-27/qwen-omni-16gb.git
cd qwen-omni-16gb
python verify_evidence.py
```

Python 3.11+ 标准库即可，预期输出：

```text
Verified four byte-identical full-vocabulary vector pairs, 256 predictions and five repeated forwards.
Historical counts / 64 contexts: {'real': 40, 'silent': 41, 'gray': 35, 'both': 32}
32 development questions x two orders; chat initialization; zero training. Not a full benchmark.
```

## 先生成一份可运行输入

```bash
python make_demo.py
```

无需额外依赖、媒体或模型下载，即可生成两秒 16kHz 单声道纯音、两张 224×224 红/蓝图片，以及 `runs/demo/spec.json`。生成目录必须是新目录，重复尝试可加 `--output-dir runs/demo-2`。

安装适合当前平台的 CUDA PyTorch 2.10.0 后：

```bash
python -m pip install -r requirements.txt
python prepare.py --spec runs/demo/spec.json --output runs/demo-inputs.pt
python infer.py --inputs runs/demo-inputs.pt --compare-stock --output runs/demo-result.json
```

预处理在 CPU 上运行，默认下载固定版本的处理器；推理还需要模型权重和 CUDA。预处理后的元信息应有 2 帧、32000 音频采样、200 有效音频特征帧。合成数据仅用于检查输入链路，不保证模型答对，也不构成准确率评测。

`--compare-stock` 比较原实现与按头分批实现；请检查结果里的数值差异和门槛是否通过。新示例已通过媒体解码测试和真实处理器验证，已有 GPU 验证范围见 [release-validation.md](docs/release-validation.md)。

## 换成自己的媒体

提供自己的 WAV 与每秒一帧、按时间排序的图片，可复制生成的 spec 或使用英文 README 的 JSON 格式：

```bash
python prepare.py --spec example.json --output runs/inputs.pt
python infer.py --inputs runs/inputs.pt --output runs/result.json
```

准备程序处理完整音频及全部提供的帧；视频抽帧由使用者完成。支持 --model 指向已有的固定版本模型目录。数值检查请使用能装下原实现的短输入，并加 --compare-stock。

媒体路径相对于 spec 文件。预处理与推理应使用同一个固定版本模型；默认模型下载可能包含未加载的语音组件，请预留上游仓库所需磁盘空间。候选概率是答案字母之间的条件概率，并不是校准置信度。

已有 4 个短输入完整词表向量精确一致、CPU 32 组算子核验及长序列选定行 FP64 检查。这不证明任意长输入与原模型逐位等价，最长原实现曾 OOM。该方案不是新注意力算法，也未证明优于 FlashAttention。

适配器只针对 transformers==5.17.0；会暂时修改进程级注意力注册表，不支持同进程并发模型调用。没有捆绑第三方音视频、权重或私人路径。实际发布验证范围见 docs/release-validation.md。

与官方方案的关系见[生态适配说明](docs/ecosystem.md)。贡献入口和小任务见 [CONTRIBUTING.md](CONTRIBUTING.md)、[路线图](docs/roadmap.md)；代码使用 [MIT 许可](LICENSE)，上游归属见 [NOTICE.md](NOTICE.md)。配套实验：[Small VLM Decision Lab](https://github.com/Cookie-27/small-vlm-decision-lab)。
