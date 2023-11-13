# nanoTransformer

`nanoTransformer` is a Python-based deep learning project focused on implementing a Transformer model for text classification tasks, particularly using the IMDB dataset. It showcases custom attention mechanisms, data preprocessing, and a training pipeline for a Transformer-based model, with a focus on clean and efficient code using PyTorch.

## Key Features

- **Efficient Attention Mechanisms**: The attention layers are implemented using `torch.einsum`, which allows for concise and efficient computation of complex tensor operations.
- **Ease of Understanding**: The code is written with readability in mind, making it accessible for those new to Transformer models and PyTorch.
- **Optimized for PyTorch**: All implementations are tailored to be efficient and seamless within the PyTorch ecosystem.

## Repository Structure

- `attention_layers.py`: Contains custom implementations of various attention mechanisms used in the Transformer model. These implementations leverage `torch.einsum` for clean and efficient tensor operations, making the code easier to understand and maintain.
- `data_utils.py`: Provides utilities for loading and preprocessing the IMDB dataset, including tokenization, vocabulary creation, and data loader preparation.
- `model.py`: Defines the architecture of the Transformer model, including Transformer blocks and the overall model structure.
- `train.py`: Script for training the Transformer model using the data loaders and model defined in `data_utils.py` and `model.py`.

## Installation

Before running the scripts, ensure you have the required dependencies:

```bash
pip install torch torchtext
```

## Usage

1. **Prepare Data**: Use `data_utils.py` to preprocess the IMDB dataset and create data loaders.

    ```python
    from data_utils import create_data_loaders
    train_loader, test_loader = create_data_loaders(batch_size=128)
    ```

2. **Define the Model**: Set up the Transformer model using `model.py`.

    ```python
    from model import Transformer
    model = Transformer(num_layers=2, model_dim=32, heads=2)
    ```

3. **Train the Model**: Train the model using `train.py`.

    ```python
    from train.py import main
    main()
    ```

## Customizing the Model

You can customize the Transformer model by adjusting parameters such as the number of layers, model dimensions, and number of attention heads in `model.py` and `train.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
