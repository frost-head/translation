# Transformer-based Translation Model

This project implements a transformer-based machine translation model using PyTorch. The model is designed to translate text from one language to another, demonstrating the power of attention mechanisms in sequence-to-sequence tasks.

## Features

- Implements a full transformer architecture for translation
- Includes positional encoding for sequence order preservation
- Uses multi-head attention mechanism
- Supports customizable model parameters (layers, heads, etc.)
- Includes a basic tokenization system

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- NumPy

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/frost-head/translation.git
   cd translation
   ```

2. Install the required packages:
   ```
   pip install torch numpy
   ```

### Downloading Pre-trained Weights

To use the pre-trained weights for this model:

1. Download the weights from [this Google Drive link](https://drive.google.com/drive/folders/1Gi3IPpbWzM4nF22e-Ok9hufUuqd5Wrlr?usp=drive_link).

2. Create a `checkpoints` folder in the root directory of the project if it doesn't exist:
   ```
   mkdir checkpoints
   ```

3. Move the downloaded weight files into the `checkpoints` folder.

### Usage

To train the model:

```python
python main.py
```

To use the pre-trained weights, make sure they are in the `checkpoints` folder before running the script.

## Model Architecture

The transformer model consists of:

- An encoder stack with multiple layers of self-attention and feed-forward networks
- A decoder stack with self-attention, encoder-decoder attention, and feed-forward networks
- Multi-head attention mechanism for capturing different aspects of the input
- Positional encoding to maintain sequence order information

## Customization

You can adjust various hyperparameters in `main.py` to experiment with different model configurations:

- `d_model`: Dimension of the model
- `nhead`: Number of attention heads
- `num_encoder_layers` and `num_decoder_layers`: Number of layers in encoder and decoder
- `dim_feedforward`: Dimension of the feed-forward network

## Contributing

Contributions to improve the model or extend its functionality are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- This implementation is inspired by the original "Attention Is All You Need" paper by Vaswani et al.
- Thanks to the PyTorch team for their excellent framework and documentation.
