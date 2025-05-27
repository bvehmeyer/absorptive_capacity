#!/usr/bin/env python3
"""
Simple Ollama Prompt to Excel Export
Sends prompts to Ollama models and exports results to Excel
Uses YAML files for models and prompts configuration
"""
import argparse
import pandas as pd
import requests
import json
import yaml
from tqdm import tqdm
import time
from datetime import datetime


class SimpleOllamaExporter:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url

    def send_prompt(self, model, prompt, temperature=0.1):
        """Send prompt to Ollama and get response"""
        url = f"{self.ollama_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048
            }
        }

        try:
            response = requests.post(url, json=data, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def process_all_combinations(self, df, models, prompts, text_column="text"):
        """Process all texts with all model-prompt combinations"""
        all_results = []

        total_combinations = len(models) * len(prompts) * len(df)
        pbar = tqdm(total=total_combinations, desc="Processing all combinations")

        for model in models:
            for prompt_name, prompt_template in prompts.items():
                for idx, row in df.iterrows():
                    # Get the text to analyze
                    text = row.get(text_column, "")

                    if not text or not isinstance(text, str):
                        pbar.update(1)
                        continue

                    # Format the prompt
                    try:
                        formatted_prompt = prompt_template.format(text=text)
                    except KeyError:
                        # If no {text} placeholder, use prompt as-is
                        formatted_prompt = prompt_template

                    # Send to model
                    start_time = time.time()
                    response = self.send_prompt(model, formatted_prompt)
                    processing_time = time.time() - start_time

                    # Compile result
                    result = {
                        'row_id': idx,
                        'model': model,
                        'prompt_name': prompt_name,
                        'input_text': text,
                        'formatted_prompt': formatted_prompt,
                        'model_response': response,
                        'processing_time_seconds': round(processing_time, 2),
                        'timestamp': datetime.now().isoformat(),
                        'text_length': len(text),
                        'response_length': len(response)
                    }

                    # Add specific required columns directly (not as original_)
                    for required_col in ['snapshot_url', 'timestamp', 'year']:
                        if required_col in row:
                            result[required_col] = row[required_col]

                    # Add other original columns with original_ prefix
                    for col, value in row.items():
                        if col not in [text_column, 'snapshot_url', 'timestamp', 'year']:
                            result[f'original_{col}'] = value

                    all_results.append(result)
                    pbar.update(1)

        pbar.close()
        return all_results


def load_yaml_config(file_path):
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Send prompts to Ollama models and export to Excel")
    parser.add_argument("--input", "-i", required=True, help="CSV file with text data")
    parser.add_argument("--output", "-o", required=True, help="Output Excel file (.xlsx)")
    parser.add_argument("--models", "-m", required=True, help="YAML file with list of models")
    parser.add_argument("--prompts", "-p", required=True, help="YAML file with prompts")
    parser.add_argument("--text-column", "-t", default="text", help="Column name containing text to analyze")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Sample size (0=all)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")

    args = parser.parse_args()

    # Load input data
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Check if text column exists
    if args.text_column not in df.columns:
        print(f"Column '{args.text_column}' not found in input file.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Sample data if requested
    if args.sample > 0:
        df = df.sample(min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} rows")

    # Load models and prompts
    models = load_yaml_config(args.models)
    prompts = load_yaml_config(args.prompts)

    if models is None or prompts is None:
        return

    if not models or not prompts:
        print("Models or prompts list is empty")
        return

    print(f"Loaded {len(models)} models: {models}")
    print(f"Loaded {len(prompts)} prompts: {list(prompts.keys())}")

    # Initialize exporter
    exporter = SimpleOllamaExporter(args.ollama_url)

    # Test connection
    try:
        test_response = requests.get(f"{args.ollama_url}/api/tags", timeout=5)
        test_response.raise_for_status()
        print(f"✓ Connected to Ollama at {args.ollama_url}")
    except Exception as e:
        print(f"✗ Cannot connect to Ollama at {args.ollama_url}: {e}")
        return

    # Calculate total combinations
    total_combinations = len(models) * len(prompts) * len(df)
    print(f"\nWill process {total_combinations} combinations:")
    print(f"  - {len(models)} models × {len(prompts)} prompts × {len(df)} texts")

    # Process all combinations
    print("\nProcessing all model-prompt combinations...")
    results = exporter.process_all_combinations(df, models, prompts, args.text_column)

    if not results:
        print("No results generated.")
        return

    # Create DataFrame and export to Excel
    results_df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        'row_id', 'model', 'prompt_name', 'snapshot_url', 'timestamp', 'year',
        'processing_time_seconds', 'text_length', 'response_length',
        'input_text', 'model_response', 'formatted_prompt'
    ]

    # Add original columns
    original_cols = [col for col in results_df.columns if col.startswith('original_')]
    column_order.extend(original_cols)

    # Reorder existing columns
    final_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[final_columns]

    # Export to Excel
    try:
        results_df.to_excel(args.output, index=False, engine='openpyxl')
        print(f"✓ Results exported to {args.output}")
        print(f"  - {len(results)} rows processed")
        print(f"  - Average processing time: {results_df['processing_time_seconds'].mean():.2f} seconds")
        print(f"  - Total processing time: {results_df['processing_time_seconds'].sum():.2f} seconds")

        # Show breakdown by model and prompt
        print(f"\nResults breakdown:")
        summary = results_df.groupby(['model', 'prompt_name']).size().reset_index(name='count')
        for _, row in summary.iterrows():
            print(f"  - {row['model']} × {row['prompt_name']}: {row['count']} results")

        # Show sample of results
        print(f"\nData statistics:")
        print(f"  - Average input text length: {results_df['text_length'].mean():.0f} chars")
        print(f"  - Average response length: {results_df['response_length'].mean():.0f} chars")

    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        # Fallback to CSV
        csv_output = args.output.replace('.xlsx', '.csv')
        results_df.to_csv(csv_output, index=False)
        print(f"✓ Fallback: Results exported to {csv_output}")


if __name__ == "__main__":
    main()