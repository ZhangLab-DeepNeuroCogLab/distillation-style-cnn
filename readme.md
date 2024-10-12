# 🎨 DISTILLATION-STYLE-CNN

*Use style-transfer to boost continual learning!*

## 🌟 Overview

This project implements the paper: [STCR: Enhancing Generalization for Continual Learning through Style Transfer and Content Replay](https://arxiv.org/abs/2211.11174)

## 🛠️ Getting Started

### Prerequisites

- 🐍 Python
- 📦 Pip

### Installation

1. Clone the repo:
   ```
   git clone https://github.com/jingcjie/distillation-style-cnn
   ```
2. Navigate to the project directory:
   ```
   cd distillation-style-cnn
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

Run the main script with:

```
python main_imagenet.py [arguments]
```

Key arguments include:
- `--batch-size`: Batch size (default: 256)
- `--epochs1`: Number of epochs for first phase (default: 70)
- `--epochs2`: Number of epochs for second phase (default: 40)
- `--new-classes`: Number of new classes (default: 10)
- `--start-classes`: Number of initial classes (default: 50)
- `--dataset`: Choose 'imagenet100' or 'imagenet'
- `--exp-name`: Experiment name
- `--no_STCR`: Disable STCR
- `--no_distill`: Disable distillation loss
- `--style_type`: Style transfer method (0: adain, 1: traditional)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

I will update this later if needed.

## 🙏 Citation

Arxiv version (waiting TNNLS bibtext):
```
@misc{shi2024unveilingtapestryinterplaygeneralization,
      title={Unveiling the Tapestry: the Interplay of Generalization and Forgetting in Continual Learning}, 
      author={Zenglin Shi and Jing Jie and Ying Sun and Joo Hwee Lim and Mengmi Zhang},
      year={2024},
      eprint={2211.11174},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2211.11174}, 
}
```