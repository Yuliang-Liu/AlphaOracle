<div align="center">
  <img src="https://i.postimg.cc/Kvk8mhqT/image-no-bg.png" width="150">
  <h1>AlphaOracle</h1>
</div>

<div align="center">
<h3>利用受人类工作流启发的深度学习框架破译甲骨文</h3>
</div>

AlphaOracle 是一套帮助专家释读甲骨文的人工智能系统。它通过甲骨文拓片与摹本分析、单字分析、单字检索以及相关文献检索四种方式，提供辅助线索并提升释读效率。

<div align="center">

![version](https://img.shields.io/badge/Version-v1.0-007acc)
![status](https://img.shields.io/badge/Status-active-00c853)
[![demo](https://img.shields.io/badge/Demo-available-ff9800)](http://vlrlabmonkey.xyz:8224/)
[![license](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English](README.md) | [中文](README_zh-CN.md)

</div>

---

## 核心功能

- 甲骨文拓片与摹本分析：覆盖字符检测、字符识别、句子切分、句内排序以及翻译为现代汉语等步骤。
- 单字分析与释读：考察甲骨文字形的历时演变，分析部件结构与构形，并生成其形态特征描述。
- 甲骨单字检索：以单个字形图像为查询，在数据库中召回所有形体相近的甲骨文字，并提供该字所在完整拓片或摹本信息。
- 关联文献检索：围绕特定甲骨文字的形体特征与语义，跨越早期传世文献与现代学术成果的双重语料库进行深度检索，呈现最相关的讨论、证据与研究结论。

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
- 覆盖甲骨文、古文字、先秦史等相关领域的大量学术论文与研究报告。更多详情请参见[原始数据来源](http://vlrlabmonkey.xyz:8224/wenxian)。

## 学术谨慎

我们致力于提供前沿的 AI 辅助能力。但由于技术与数据的限制，输出结果仍可能存在偏差或误释。如需在正式出版物中引用本系统产生的见解，请完善注明出处，并进行必要的学术核查。

## 网站演示


https://github.com/user-attachments/assets/70e2c3d0-6da2-45af-b03e-76ae5ea8a0e7




## 许可协议

[MIT LICENSE](LICENSE)

---

© 2025 AlphaOracle 项目组
