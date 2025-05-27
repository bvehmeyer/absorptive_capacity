# Simple Ollama to Excel Exporter

A streamlined tool to send prompts to Ollama models and export raw results to Excel.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas requests tqdm openpyxl PyYAML
```

### 2. Install Ollama

**Docker (Recommended):**
```bash
# CPU only
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# With GPU support
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**Direct Installation:**
- **Linux/WSL:** `curl -fsSL https://ollama.com/install.sh | sh && ollama serve`
- **macOS:** Download from [ollama.com](https://ollama.com/download/mac)
- **Windows:** Download installer from [ollama.com](https://ollama.com/download/windows)

### 3. Download Models
```bash
ollama pull llama3.2:3b    
ollama pull mistral:7b     
ollama pull qwen2.5:7b   

### 4. Adjust Configuration Files

**models.yaml:**
```yaml
- llama3.2:3b
- mistral:7b
- qwen2.5:7b
```

**prompts.yaml:**
```yaml
absorptive_analysis: |
  Rate this company's absorptive capacity
  1. Identification (recognizing external knowledge)
  2. Assimilation (understanding external knowledge) 
  3. Transformation (combining knowledge)
  4. Exploitation (commercial utilization)
  
  Company text: {text}

simple_classification: |
  Classify this company's main business focus: {text}
```

## 📊 Usage

### Basic Command - for the entire dataset
```bash
python ollama_model_test.py \
  --input data/web_texts.csv \
  --output results.xlsx \
  --models models.yaml \
  --prompts prompts.yaml
```

### With Sample Size - for small experiments
```bash
python ollama_model_test.py \
  --input data/web_texts.csv \
  --output results.xlsx \
  --models models.yaml \
  --prompts prompts.yaml \
  --sample 10
```

## 📁 Input Requirements

Your CSV file should have:
- `text` column (company website content)
- `snapshot_url` column (website URL)
- `timestamp` column (when scraped)
- `year` column (year of data)

## 📊 Excel Output

The generated Excel file contains:

| Column | Description |
|--------|-------------|
| `row_id` | Row number from input |
| `model` | Model used for analysis |
| `prompt_name` | Prompt template used |
| `snapshot_url` | Website URL |
| `timestamp` | Scraping timestamp |
| `year` | Year of data |
| `processing_time_seconds` | Time taken per request |
| `text_length` | Length of input text |
| `response_length` | Length of model response |
| `input_text` | Original company text |
| `model_response` | **Raw model response** |
| `formatted_prompt` | Complete prompt sent to model |

## 🛠️ Command Line Options

```bash
python ollama_model_test.py \
  --input data/web_texts.csv        # Input CSV file
  --output results.xlsx             # Output Excel file
  --models models.yaml              # YAML file with model list
  --prompts prompts.yaml            # YAML file with prompts
  --sample 10                       # Optional: limit to N rows from input file
  --text-column text                # Optional: specify text column name
  --ollama-url http://localhost:11434  # Optional: custom Ollama URL
```

## 🔧 Troubleshooting

**Connection Error:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

**Model Not Found:**
```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2:3b
```