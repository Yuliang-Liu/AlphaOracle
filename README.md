<div align="center">
  <img src="https://i.postimg.cc/BQyDg2jN/image-no-bg.png" width="150">
  <h1>AlphaOracle</h1>
</div>

### Towards a foundational, AI‑assisted workflow for Oracle Bone Script decipherment.

AlphaOracle is an artificial intelligence system that assists experts in deciphering oracle bone script. It provides auxiliary clues for experts and improves interpretation efficiency through four methods: analysis of rubbings and facsimiles, analysis of individual oracle bone characters, retrieval of individual oracle bone characters, and retrieval of related literature.

<div align="center">

![version](https://img.shields.io/badge/Version-v1.0-007acc)
![status](https://img.shields.io/badge/Status-active-00c853)
[![demo](https://img.shields.io/badge/Demo-available-ff9800)](http://vlrlabmonkey.xyz:8224/)
[![license](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English](README.md) | [中文](README_zh-CN.md)

</div>

---

## Key Features

- Analysis of oracle bone rubbings and transcriptions: involving character detection, character recognition, sentence segmentation, intra-sentence ordering, and translation into Modern Chinese.
- Analysis and Decipherment of Individual Oracle Bone Characters: investigating the diachronic evolution of their glyph forms, analyzing their component structures and configurations, and generating descriptions of their morphological features. 
- Retrieval for Oracle Bone Characters: using a single character image as a query, the system retrieves all characters with similar glyph shapes from the database. For each result, it provides the corresponding complete rubbing or transcription in which the character is found.
- Associative Literature Retrieval: based on the glyphic form and semantic meaning of a specific oracle bone character, the system performs a deep search across a dual corpus—encompassing early transmitted Chinese texts (from the pre-Qin and Han dynasties) and modern scholarly literature—to discover and present the most relevant discussions, evidence, and research findings.

## Tutorial

This project includes a Jupyter Notebook that demonstrates the usage of each API in AlphaOracle. It begins by defining common utilities and wrapper functions, and then demonstrates the input and output formats for each task in separate code blocks.

[View the Tutorial Notebook](example/demo.ipynb)

## Data Sources

This project draws upon a comprehensive collection of textual and inscriptional resources, including:

### Oracle Bone Inscription Databases:
- **Rubbings and Transcriptions**: 
  - 《Jia Gu Wen He Ji》 
  - 《Jia Gu Wen Mo Ben Da Xi》
  - 《Jia Gu Wen Jiao Shi Zong Ji》
- **Glyph Information**:
  - 《Xin Jia Gu Wen Bian》
  - 《Jia Gu Wen Liu Wei Shu Zi Ma Jian Suo Zi Ku》
  - 《Xi Zhou Jin Wen Zi Bian》
  - 《Chun Qiu Wen Zi Zi Xing Biao》
  - 《Zhan Guo Wen Zi Zi Xing Biao》
  - 《Shuo Wen Jie Zi》
  - Yin Qi Wen Yuan (website)
  - Guo Xue Da Shi (website)
- **Interpretive Resources**:
  - 《Gu Wen Zi Gu Lin》
  - 《Jia Gu Wen Gu Lin》
- **Sentence Translation**:
  - 《Jia Gu Wen Jing Cui Shi Yi》

### Pre-Qin and Han Transmitted Texts:
- 《Han Shu》, 《Shi Ji》, 《Zuo Zhuan》, 《Huangdi Neijing》, 《Zhanguo Ce》, 《Huainanzi》, 《Han Feizi》, 《Li Ji》, 《Lu Shi Chunqiu》, 《Guo Yu》, 《Yi Li》, 《Zhuangzi》, 《Mozi》, 《Zhou Li》, 《Mengzi》, 《Shan Hai Jing》, 《Shang Shu》, 《Xunzi》, 《Lunyu》, 《Yizhuan》, 《Sun Bin Bingfa》, 《Laozi》, 《Sunzi Bingfa》, 《Wuzi》

### Modern Scholarly Literature:
A vast collection of modern academic papers and research reports focusing on oracle bone script, paleography, ancient Chinese history, and related fields. For a detailed list, please refer to the [original data source](http://vlrlabmonkey.xyz:8224/wenxian).

## Academic Rigor

We are committed to providing cutting-edge AI assistance. However, due to inherent technological and data limitations, the output may contain discrepancies or misinterpretations. If you reference insights from this system in any published work, please include an appropriate acknowledgement and conduct your own final academic verification.

## License

[MIT LICENSE](LICENSE).

---

© 2025 AlphaOracle Project Team
