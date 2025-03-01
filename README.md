# Urdu Poem Generator

This project includes web scraping, text generation using deep learning, and a Streamlit-based user interface for generating Urdu poetry.

## Installation

To run this project, install the required dependencies:

```sh
pip install streamlit tensorflow numpy pandas pickle requests beautifulsoup4 time os
```

### Additional Dependencies

Ensure you have the necessary system dependencies installed for TensorFlow and web scraping.

#### Install System Dependencies (Optional)
- **Windows**: Install required packages using `pip`.
- **MacOS/Linux**: Ensure you have `pip` and system libraries for TensorFlow.

## Modules

### 1. **Application & Model Integration**
The following libraries are used for the Streamlit-based UI and model integration:

- `streamlit`: Web UI framework
- `tensorflow`: Deep learning model framework
- `numpy`: Numerical operations
- `pickle`: Model serialization

### 2. **Web Scraper**
Used for scraping poetry data from [Rekhta](https://www.rekhta.org):

- `requests`: Fetch web pages
- `beautifulsoup4`: Parse HTML
- `pandas`: Store and process scraped data
- `time`: Handling delays in requests
- `os`: File operations

### 3. **Model Training**
The deep learning model utilizes:

- `tensorflow.keras.preprocessing.text.Tokenizer`: Tokenization
- `tensorflow.keras.models.Sequential`: Model architecture
- `tensorflow.keras.layers.Embedding, LSTM, Dense, Dropout`: Layers for training
- `tensorflow.keras.optimizers.Adam`: Optimizer
- `pickle`: Model saving

## Usage

Run the Streamlit app with:

```sh
streamlit run app.py
```

## License

This project is open-source and available under the MIT License.

