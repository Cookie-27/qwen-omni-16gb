# 在 16GB 显卡上运行 Qwen Omni 候选评分

[English](README.md)

这是针对 Qwen2.5-Omni-3B Thinker 的小型推理示例。通过把文本注意力按头分批调用原 SDPA，降低某些回退实现的峰值显存；每个头保留完整序列和对应 KV 头，音频与视觉注意力仍使用原实现。

历史 RTX 5080 实测：完整约 40–60 秒音频、全部选定的每秒一帧图像，最长 5,521 token；PyTorch 分配峰值 10.202 GiB，整卡采样峰值 12,001 MiB。加载的 Thinker 实际为约 4.703B 参数。

**范围是单次前向、末位候选答案评分。** 不包含语音生成、任意长文本生成或训练的显存承诺。

无需 GPU 或额外依赖即可核验历史结果：

```bash
python verify_evidence.py
```

提供自己的 WAV 与按时间排序的帧图片，依照英文 README 编写输入 JSON，安装对应 CUDA PyTorch 和 requirements.txt 后运行：

```bash
python prepare.py --spec example.json --output runs/inputs.pt
python infer.py --inputs runs/inputs.pt --output runs/result.json
```

准备程序处理完整音频及全部提供的帧；视频抽帧由使用者完成。支持 --model 指向已有的固定版本模型目录。数值检查请使用能装下原实现的短输入，并加 --compare-stock。

已有 4 个短输入完整词表向量精确一致、CPU 32 组算子核验及长序列选定行 FP64 检查。这不证明任意长输入与原模型逐位等价，最长原实现曾 OOM。该方案不是新注意力算法，也未证明优于 FlashAttention。

适配器只针对 transformers==5.17.0；会暂时修改进程级注意力注册表，不支持同进程并发模型调用。没有捆绑第三方音视频、权重或私人路径。实际发布验证范围见 docs/release-validation.md。
