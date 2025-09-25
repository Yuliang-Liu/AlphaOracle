<div align="center">
  <img src="https://i.postimg.cc/7LhQghMy/image-no-bg-smooth.png" width="150">
  <h1>AlphaOracle</h1>
</div>

### Towards a foundational, AI‑assisted workflow for Oracle Bone Script decipherment.

AlphaOracle is an artificial intelligence system that assists experts in deciphering oracle bone script. It provides auxiliary clues for experts and improves interpretation efficiency through four methods: analysis of rubbings and facsimiles, analysis of individual oracle bone characters, retrieval of individual oracle bone characters, and retrieval of related literature.

![version](https://img.shields.io/badge/Version-v1.0-007acc)
![status](https://img.shields.io/badge/Status-active-00c853)
[![demo](https://img.shields.io/badge/Demo-available-ff9800)](http://vlrlabmonkey.xyz:8224/)
[![license](https://img.shields.io/badge/License-MIT-green)](LICENSE)

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
  - 《甲骨文合集》 
  - 《甲骨文摹本大系》
  - 《甲骨文校释总集》
- **Glyph Information**:
  - 《新甲骨文编》
  - 《甲骨文六位数字码检索字库》
  - 《西周金文字编》
  - 《春秋文字字形表》
  - 《战国文字字形表》
  - 《说文解字》
  - 殷契文渊 (website)
  - 国学大师 (website)
- **Interpretive Resources**:
  - 《古文字诂林》
  - 《甲骨文诂林》
- **Sentence Translation**:
  - 《甲骨文精粹释译》

### Pre-Qin and Han Transmitted Texts:
- 《汉书》, 《史记》, 《左传》, 《黄帝内经》, 《战国策》, 《淮南子》, 《韩非子》, 《礼记》, 《吕氏春秋》, 《国语》, 《仪礼》, 《庄子》, 《墨子》, 《周礼》, 《孟子》, 《山海经》, 《尚书》, 《荀子》, 《论语》, 《易传》, 《孙膑兵法》, 《老子》, 《孙子兵法》, 《吴子》

### Modern Scholarly Literature:
A vast collection of modern academic papers and research reports focusing on oracle bone script, paleography, ancient Chinese history, and related fields. For a detailed list, please refer to the [original data source](http.vlrlabmonkey.xyz:8224/wenxian).

## Academic Rigor

We are committed to providing cutting-edge AI assistance. However, due to inherent technological and data limitations, the output may contain discrepancies or misinterpretations. If you reference insights from this system in any published work, please include an appropriate acknowledgement and conduct your own final academic verification.

## License

[MIT LICENSE](LICENSE).

---

© 2025 AlphaOracle Project Team
