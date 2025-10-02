
# ViLBias — Detecting and Reasoning about Bias in Multimodal Content

VILBias is a framework designed to detect and analyze bias in multimodal contexts, integrating linguistic and visual cues to improve transparency, accountability, and fairness in media representation.

[![arXiv](https://img.shields.io/badge/arXiv-2412.17052-b31b1b.svg)](https://arxiv.org/abs/2412.17052)
[![Cite (BibTeX)](https://img.shields.io/badge/Cite-BibTeX-1f425f.svg)](#-citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 

<!-- Teaser -->
<p align="center">
  <img src="docs/VILBias.png" alt="ViLBias data construction, annotation, and evaluation overview." width="85%">
</p>

## Abstract
**ViLBias** is a VQA-style benchmark and framework for detecting and *reasoning* about bias in news media using text–image pairs.  
The dataset contains **~40k** article–image examples with binary labels (biased / not biased) and short rationales produced via a hybrid **LLM-as-annotator** pipeline with hierarchical voting and **human-in-the-loop** validation. We evaluate SLMs, LLMs, and VLMs on both closed-ended classification and open-ended reasoning, and observe steady gains from incorporating visual cues and instruction tuning. ViLBias provides scalable data, baselines, and metrics for multimodal bias detection and rationale quality. :contentReference[oaicite:0]{index=0}

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📚 Citation

If you use VILBias or wish  to use any of this work, please cite:

```bibtex
@misc{raza2025vilbiasdetectingreasoningbias,
      title={ViLBias: Detecting and Reasoning about Bias in Multimodal Content}, 
      author={Shaina Raza and Caesar Saleh and Azib Farooq and Emrul Hasan and Franklin Ogidi and Maximus Powers and Veronica Chatrath and Marcelo Lotif and Karanpal Sekhon and Roya Javadi and Haad Zahid and Anam Zahid and Vahid Reza Khazaie and Zhenyu Yu},
      year={2025},
      eprint={2412.17052},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.17052}, 
}

```

## Contact

For general inquiries or image data, please contact:
- **Shaina Raza**  
  Email: [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai)
