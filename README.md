# MetaphorsOfMachine

MetaphorsOfMachine is a research project for analyzing how users metaphorically frame AI in conversations. The project uses natural language processing and machine learning techniques to extract semantic frames, detect metaphors, and analyze patterns across demographic dimensions.

## Project Overview

This project analyzes conversations from the WildChat dataset to understand how people conceptualize and talk about AI, focusing on:

- Extracting AI references from user messages
- Detecting metaphorical language in these references
- Identifying semantic frames that structure how users think about AI
- Analyzing patterns across countries and reference types
- Generating visualizations and comprehensive reports

## Installation

### Requirements

To install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn tqdm nltk transformers torch torchvision sklearn datasets plotly wordcloud joblib
```

For GPU acceleration (recommended for frame extraction):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Download required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Directory Structure

Ensure you have the following directory structure:

```
MetaphorsOfMachine/
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
├── src/
│   ├── data.py
│   ├── frames.py
│   ├── clustering.py
│   └── analysis.py
├── main.py
├── config.py
├── analyze_only.py
└── run_pipeline.sh
```

You can create the necessary directories using:

```bash
mkdir -p data/raw data/processed results/figures results/tables results/models
```

## Configuration

Modify `config.py` to customize the project settings:

- `USE_SAMPLE`: Set to `True` to use a sample of the dataset, `False` for the full dataset
- `SAMPLE_SIZE`: Number of conversations to sample
- `MIN_COUNTRY_COUNT`: Minimum entries for a country to be included in analysis
- `TOP_FRAMES`: Number of top frames to analyze
- `DEVICE`: Set to 'cuda' for GPU, 'cpu' for CPU, or None for auto-selection

Example configuration:

```python
# in config.py
USE_SAMPLE = True
SAMPLE_SIZE = 5000
DEVICE = 'cuda'  # Use 'cpu' if no GPU available
```

## Running the Pipeline

### Full Pipeline

To run the complete pipeline (data loading, preprocessing, frame extraction, clustering, and analysis):

```bash
python main.py
```

Command-line options:

```bash
python main.py --full  # Use full dataset
python main.py --sample-size 10000  # Use 10,000 samples
python main.py --force-reload  # Force reload of dataset
python main.py --skip-frames  # Skip frame extraction
python main.py --skip-clustering  # Skip clustering
python main.py --skip-analysis  # Skip demographic analysis
python main.py --report-only  # Only generate report from existing data
```

### Analysis Only

If you already have processed data and just want to run the analysis:

```bash
python analyze_only.py
```

### Running on a Cluster

For running on a SLURM-based cluster, modify `run_pipeline.sh` with your account details and submit:

```bash
sbatch run_pipeline.sh
```

## Project Components

### Data Processing

The `WildChatDataProcessor` class loads, preprocesses, and extracts AI references from the WildChat dataset.

### Frame Extraction

The `FrameExtractor` class uses transformer models to:
- Detect metaphorical language
- Identify novel metaphors
- Extract semantic frames

### Clustering

The `ClusterAnalyzer` class:
- Reduces dimensions of sentence embeddings
- Clusters similar frame patterns
- Visualizes cluster distributions

### Analysis

The `DemographicAnalyzer` class:
- Analyzes patterns across geographic regions
- Compares direct vs. indirect AI references
- Examines frame co-occurrence patterns
- Generates visualizations and reports

## Output

The pipeline generates:

- **Tables**: JSON and Parquet files with analysis results
- **Figures**: Visualizations of patterns (PNG format)
- **Report**: A comprehensive Markdown report summarizing findings

Results are saved in the `results/` directory.