import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import re
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

def format_content(string):
    # Remove HTML tags
    text = BeautifulSoup(string, "html.parser").get_text()
    # Remove weird formatting where a full stop is represented by: .,
    text = re.sub(r'\.,', '.', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_summary(soup, tokenizer, model):
    content = soup.find_all('p')
    formatted_content = format_content(str(content))
    # Encode the input text into PyTorch tensors
    inputs = tokenizer.encode(formatted_content, return_tensors="pt", truncation=True, max_length=1024)
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    # Generate summary with set length constraints
    summary_ids = model.generate(inputs, min_length=100, max_length=300, do_sample=False, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_caption(soup, publisher):
    content = ""
    if publisher == "metro":
        content = soup.find("figcaption")
        content = content.get_text() if content else "No Caption"
        # Credits is contained in parenthesis in image captions from Metro
        content = re.sub(r'\s*\([^)]+\)', '', content)
    elif publisher == "mail":
        content = soup.find("p", class_="imageCaption")
        content = content.get_text() if content else "No Caption"
    elif publisher == "guardian":
        content = soup.find("figcaption").find("span", class_="dcr-1qvd3m6")
        content = content.get_text() if content else "No Caption"
    elif publisher == "sun":
        content = soup.find("figcaption").find("span", class_="article__media-span")
        content = content.get_text() if content else "No Caption"
    else:
        content = 'unknown'
    
    caption = content.strip()
    return caption

def retrieve_content(url, tokenizer, model, publisher):
    try:
        response = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.10 Safari/605.1.1"
        })
        if response.status_code != 200:
            return None, None, None
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No Title"
        summary = generate_summary(soup, tokenizer, model)
        caption = generate_caption(soup, publisher)
        return title, summary, caption
    except Exception as e:
        print(f"Failed to retrieve content for URL: {url} | Error: {e}")
        return None, None, None

def create_processed_dataset(df, rows, tokenizer, model):
    dataset = []
    
    for i in range(rows):
        publisher = df.loc[i, 'Paper']
        if publisher in {"times", "telegraph"}:
            continue
        title, summary, caption = retrieve_content(df.loc[i, 'Link'], tokenizer, model, publisher)
        if title is None or summary is None or caption is None:
            print(f"Failed to retrieve content for dataset item {i + 1}")
            continue
        datasetItem = {
            "ID": i + 1,
            "Metadata": { 
                "NewsArticle_Title": title,
                "NewsArticle_Publisher": publisher,
                "NewsArticle_Link": df.loc[i, 'Link'],
                "Datetime_Extracted": df.loc[i, 'ExtractDatetime']
            },
            "NewsArticle_Summary": summary,
            "NewsArticle_Image_Caption": caption
        }
        print(f"Processed dataset item {i + 1}")
        dataset.append(datasetItem)

    return dataset

def export_to_json(dataset):
    with open('processedData.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

def main():
    # Extract data from the raw dataset
    df = pd.read_csv('Raw_Dataset.csv', encoding='utf-8')
    rows = df.shape[0]
    
    # Initialise the tokenizer and model using PyTorch
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # Prepare the data to fit the JSON format
    dataset = create_processed_dataset(df, rows, tokenizer, model)
    
    # Export the processed data to a JSON file
    export_to_json(dataset)

if __name__ == '__main__':
    main()