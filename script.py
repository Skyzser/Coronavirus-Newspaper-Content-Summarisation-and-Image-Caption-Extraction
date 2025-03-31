import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline

def format_content(string):
    # Remove HTML tags
    text = BeautifulSoup(string, "html.parser").get_text()
    # Remove weird formatting where a full stop is represented by: .,
    text = re.sub(r'\.,', '.', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_summary(soup, summariser):
    content = soup.find_all('p')
    formatted_content = format_content(str(content))
    summary = summariser(formatted_content, min_length=150, max_length=300, do_sample=False)
    return summary[0]['summary_text'] if summary else "No Summary"

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
    else:
        content = 'unknown'
    
    caption = content.strip()
    return caption

def retrieve_content(url, summariser, publisher):
    try:
        response = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0"  # Mimic a browser
        })
        if response.status_code != 200:
            return None, None, None
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.title.string if soup.title else "No Title", generate_summary(soup, summariser), generate_caption(soup, publisher)
    except Exception as e:
        print(f"Failed to retrieve content for URL: {url} | Error: {e}")
        return None, None, None

def create_processed_dataset(df, rows, summariser):
    dataset = []

    def process_row(i):
        publisher = df.loc[i, 'Paper']
        if publisher in {"times", "telegraph"}:
            return None
        title, summary, caption = retrieve_content(df.loc[i, 'Link'], summariser, publisher)
        if title is None or summary is None or caption is None:
            print(f"Failed to retrieve content for dataset item {i + 1}")
            return None
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
        return datasetItem

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_row, range(rows)))

    dataset = [item for item in results if item is not None]
    return dataset

def export_to_json(dataset):
    with open('processedData.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

def main():
    # Extract data from the raw dataset
    df = pd.read_csv('Raw_Dataset.csv', encoding='utf-8')
    rows = df.shape[0]
    
    # Initialise the summariser
    summariser = pipeline("summarization", model="facebook/bart-large-cnn")

    # Prepare the data to fit the JSON format
    dataset = create_processed_dataset(df, rows, summariser)
    
    # Export the processed data to a JSON file
    export_to_json(dataset)

if __name__ == '__main__':
    main()