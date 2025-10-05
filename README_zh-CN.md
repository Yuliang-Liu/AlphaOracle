<div align="center">
  <img src="https://v1.ax1x.com/2025/10/03/EIkNeV.png" width="200">
</div>

<div align="center">
<h3>利用受人类工作流启发的深度学习框架破译甲骨文</h3>
  <p>刘宇梁<sup>*,†</sup>, 关海苏<sup>*</sup>, 王鹏杰<sup>*</sup>, 王新宇<sup>*</sup>, 万金鹏, 张凯乐, 郑翰东, 刘星辰, 邝哲彬, 杨欢鑫, 李邦, 刘永革, 金连文<sup>†</sup>, 白翔<sup>†</sup></p>
<p><sup>*</sup>同等贡献, <sup>†</sup>通讯作者</p>
</div>

我们引入了一种受人类工作流程启发的方法 AlphaOracle，它可以加速甲骨文学术研究，同时保持透明度和严谨性。

<div align="center">

![version](https://img.shields.io/badge/Version-v1.0-007acc)
![status](https://img.shields.io/badge/Status-active-00c853)
[![demo](https://img.shields.io/badge/Demo-available-ff9800)](http://www.alphaoracle.cn:8224/)
[![license](https://img.shields.io/badge/License-Apache-green)](LICENSE)

[English](README.md) | [中文](README_zh-CN.md)

</div>

---

<details open><summary>💡 我还有其他您可能感兴趣的项目 ✨。</summary><p>
    
> [**Deciphering Oracle Bone Language with Diffusion Models**](https://arxiv.org/abs/2406.00684) <br>
> 管海粟, 杨焕鑫, 王欣雨, 韩胜伟, 刘永革, 金连文, 白翔, 刘禹良 <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/guanhaisu/OBSD) [![arXiv](https://img.shields.io/badge/Arxiv-2406.00684-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.00684) <br>
    
> [**Puzzle Pieces Picker: Deciphering Ancient Chinese Characters with Radical Reconstruction**](https://arxiv.org/abs/2406.03019) <br>
> 王鹏杰, 张凯乐, 王欣雨, 韩胜伟, 刘永革, 金连文, 白翔, 刘禹良 <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Pengjie-W/Puzzle-Pieces-Picker) [![arXiv](https://img.shields.io/badge/Arxiv-2406.03019-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.03019) <br>

> [**An open dataset for oracle bone character recognition and decipherment**](https://www.nature.com/articles/s41597-024-03807-x) <br>
> 王鹏杰, 张凯乐, 王欣雨, 韩胜伟, 刘永革, 万金鹏, 管海粟, 匡嚞玢, 金连文, 白翔, 刘禹良 <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Pengjie-W/HUST-OBC) [![arXiv](https://img.shields.io/badge/Scientific_Data-s41597.024.03807-gren.svg?)](https://www.nature.com/articles/s41597-024-03807-x) <br>

> [**An open dataset for the evolution of oracle bone characters: EVOBC**](https://arxiv.org/abs/2401.12467) <br>
> 管海粟, 万金鹏, 刘禹良, 王鹏杰, 张凯乐, 匡嚞玢, 王欣雨, 白翔, 金连文 <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RomanticGodVAN/character-Evolution-Dataset) [![arXiv](https://img.shields.io/badge/Arxiv-2401.12467-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2401.12467) <br>

## 📰 新闻

* `[2025.10.5]` 我们发布了 AlphaOracle 的论文和[演示](http://www.alphaoracle.cn:8224/)。

</p></details>

## 核心功能

- 甲骨文拓片与摹本分析：覆盖字符检测、字符识别、句子切分、句内排序以及翻译为现代汉语等步骤。
<div align="center">
  <img src="figures/zh/key1.png" width="600">
</div>
- 单字分析与释读：考察甲骨文字形的历时演变，分析部件结构与构形，并生成其形态特征描述。
<div align="center">
  <img src="figures/zh/key2.png" width="600">
</div>
- 甲骨单字检索：以单个字形图像为查询，在数据库中召回所有形体相近的甲骨文字，并提供该字所在完整拓片或摹本信息。
<div align="center">
  <img src="figures/zh/key3.png" width="600">
</div>
- 关联文献检索：围绕特定甲骨文字的形体特征与语义，跨越早期传世文献与现代学术成果的双重语料库进行深度检索，呈现最相关的讨论、证据与研究结论。
<div align="center">
  <img src="figures/zh/key4.png" width="600">
</div>
<div align="center">
  <img src="figures/en/1.jpg" width="800">
</div>

## 使用教程

项目附带 Jupyter Notebook，逐步演示 AlphaOracle 每个 API 的调用方式。Notebook 先定义常用工具与封装函数，再为每项任务分别展示输入输出示例。

[查看示例 Notebook](example/demo.ipynb)


## 数据来源

本项目汇聚了丰富的甲骨文资料与相关文献，包括：

### 甲骨文数据库：
- **拓片与摹本**：
  - 《甲骨文合集》
  - 《甲骨文摹本大系》
  - 《甲骨文校释总集》
- **字形信息**：
  - 《新甲骨文编》
  - 《甲骨文六位数字码检索字库》
  - 《西周金文字编》
  - 《春秋文字字形表》
  - 《战国文字字形表》
  - 《说文解字》
  - 殷契文渊（网站）
  - 国学大师（网站）
- **释读参考**：
  - 《古文字诂林》
  - 《甲骨文诂林》
- **句子翻译**：
  - 《甲骨文精粹释译》

### 先秦两汉传世文献：
- 《汉书》, 《史记》, 《左传》, 《黄帝内经》, 《战国策》, 《淮南子》, 《韩非子》, 《礼记》, 《吕氏春秋》, 《国语》, 《仪礼》, 《庄子》, 《墨子》, 《周礼》, 《孟子》, 《山海经》, 《尚书》, 《荀子》, 《论语》, 《易传》, 《孙膑兵法》, 《老子》, 《孙子兵法》, 《吴子》

### 现代学术文献：
覆盖甲骨文、古文字、先秦史等相关领域的大量学术论文与研究报告。更多详情请参见[原始数据来源](http://www.alphaoracle.cn:8224/wenxian)。

## 视频演示

https://github.com/user-attachments/assets/69ea0636-37bb-4fef-a0f6-80a27fa21105

我们提供了网站 [AlphaOracle](http://www.alphaoracle.cn:8224) 以便快速体验和使用。

## 待办事项

- [x] 演示
- [ ] API
- [ ] 模型权重
- [ ] 推理代码
- [ ] 完整的释读流程
- [ ] 训练代码

## 学术声明

我们致力于提供前沿的 AI 辅助。然而，由于固有的技术和数据限制，输出可能包含差异或误解。如果您在任何已发表的作品中引用本系统的见解，请附上适当的致谢，并进行您自己的最终学术验证。

## 许可协议

[Apache License](LICENSE)

## ✏️ 引用

```BibTeX
@article{liu2025oracle,
  title={Oracle bone script decipherment via human-workflow-inspired deep learning},
  author={Yuliang Liu, Haisu Guan, PengJie Wang, Xinyu Wang, Jinpeng Wan, Kaile Zhang, Handong Zheng, Xingchen Liu, Zhebin Kuang, Huanxin Yang, Bang Li, Yonge Liu, Lianwen Jin and Xiang Bai},
  year={2025}
}
```

## 🤝 贡献者

<a href="https://github.com/Yuliang-Liu/AlphaOracle/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Yuliang-Liu/AlphaOracle" />
</a>

© 2025 AlphaOracle 项目组
