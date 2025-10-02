
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

Detecting bias in multimodal news requires models that **reason over text–image pairs**, not just classify text. In response, we present **ViLBias**, a **VQA-style benchmark and framework** for detecting and reasoning about bias in multimodal news. The dataset comprises **40,945** text–image pairs from diverse outlets, each annotated with a **bias label and concise rationale** using a **two-stage LLM-as-annotator pipeline** with **hierarchical majority voting** and **human-in-the-loop** validation. We evaluate **Small Language Models (SLMs)**, **Large Language Models (LLMs)**, and **Vision–Language Models (VLMs)** across **closed-ended classification** and **open-ended reasoning (oVQA)**, and compare **parameter-efficient tuning** strategies. Results show that **incorporating images** alongside text improves detection accuracy by **3–5%**, and that **LLMs/VLMs** better capture subtle framing and text–image inconsistencies than SLMs. **Parameter-efficient methods (LoRA/QLoRA/Adapters)** recover **97–99%** of full fine-tuning performance with **<5% trainable parameters**. For oVQA, **reasoning accuracy = 52–79%** and **faithfulness = 68–89%**, both improved by **instruction tuning**; **closed accuracy correlates strongly with reasoning (r = 0.91)**. ViLBias offers a **scalable benchmark** and **strong baselines** for multimodal bias detection and rationale quality.



## Contribution

We welcome contributions to improve VILBias. Please follow these steps to contribute:
1. Fork this repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git commit -m "Description of changes"
   git push origin feature-name
   ```
4. Create a pull request.

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
