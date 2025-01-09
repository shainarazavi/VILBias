
# ViLBias: A Comprehensive Framework for Bias Detection through Linguistic and Visual Cues , presenting Annotation Strategies, Evaluation, and Key Challenges

VILBias is a framework designed to detect and analyze bias in multimodal contexts, integrating linguistic and visual cues to improve transparency, accountability, and fairness in media representation.

## Overview

This repository contains code for:
- **SLMs (Small Language Models):** Lightweight models optimized for efficiency and suitable for resource-constrained environments.
- **LLMs (Large Language Models):** Advanced models designed for deeper linguistic understanding and nuanced bias detection in text and multimodal data.
- **Multimodal Analysis:** Techniques for detecting bias through the combination of text and image inputs, addressing issues like representational imbalance, framing, and selective presentation.

## Features

- **Text-Based Bias Detection:** Identify rhetorical techniques such as emotional appeal, selective presentation, and framing.
- **Text + Image Analysis:** Evaluate how textual and visual elements interact to reinforce or contradict narratives.
- **Customizable Evaluation Criteria:** Adjust analysis parameters to suit specific applications or domains.

## Requirements

- Python 3.8 or above
- Required libraries: `torch`, `transformers`, `opencv`, `scikit-learn`, and `pandas`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/shainarazavi/VILBias.git
   cd VILBias
   ```

2. Run a text-based bias analysis:
   ```bash
   python analyze_text.py --input example_text.txt
   ```

3. Perform multimodal analysis (text + image):
   ```bash
   python analyze_multimodal.py --text example_text.txt --image example_image.jpg
   ```

4. Customize evaluation parameters in `config.json` to suit your dataset.

## Data Availability

The dataset used in this study is not publicly available due to privacy considerations but can be provided upon reasonable request. Please contact [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai) for access.

## Models

### Small Language Models (SLMs)
SLMs provide lightweight solutions for bias detection, enabling applications in low-resource settings. These models are trained for efficiency and quick inference.

### Large Language Models (LLMs)
LLMs deliver advanced reasoning capabilities, handling nuanced linguistic and multimodal contexts for high-fidelity bias detection.

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

## Contact

For inquiries, please contact:
- **Shaina Raza**  
  Email: [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai)
```

