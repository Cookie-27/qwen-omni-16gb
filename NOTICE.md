# Sources and attribution

Original adapter, release tooling and exported experimental measurements: Cookie-27, MIT license (LICENSE).

The implementation calls the installed [Hugging Face Transformers](https://github.com/huggingface/transformers) SDPA integration and [PyTorch](https://pytorch.org/). Upstream source and weights are not vendored; their licenses apply separately.

Model: [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B), revision f75b40e3da2003cdd6e1829b1f420ca70797c34e. Only the Thinker is loaded; it is conversational initialization, not a raw pretrained Base. Obtain model weights from the official publisher under its model license.

Historical runtime media and labels came from [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA), code/annotation revision cc8809514a2ee3d7ae81a3021c15e7c0679170d7, data/json_update. Attribution: Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen and Di Hu, Learning to Answer Questions in Dynamic Audio-Visual Scenarios, CVPR 2022. The repository declares MIT for its software; this release does not redistribute its videos, audio or images or assert that code licensing covers third-party media.

Short numerical anchors used [AV-Odyssey](https://huggingface.co/datasets/AV-Odyssey/AV_Odyssey_Bench_LMMs_Eval), fixed distribution revision ef40fb2222c31142a87cc564633c27e9f8c63b11. No question text or media is included; only model-produced numeric vectors and measurements are exported.

The repository is independent experimental tooling, not an official Qwen, PyTorch or Transformers project. MIT here does not relicense upstream models, data or dependencies.
