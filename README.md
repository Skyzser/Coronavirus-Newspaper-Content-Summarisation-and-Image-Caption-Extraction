# Coronavirus Newspaper Content Summarisation and Image Caption Extraction

This project, which is done as part of my M.Sc's Text Analytics module, processes a dataset of COVID-19 related newspaper articles, retrieving each article's content, generating a summary using NLP techniques, and extracting the image caption from the article's webpage. The final structured data is exported to a JSON file.

## Dataset

Dataset used: [Coronavirus Newspaper Classification](https://www.kaggle.com/code/jwallib/coronavirus-newspaper-classification/notebook)

- Contains the data for the news article publisher, the link to the news article, and the date that the news article was extracted for the dataset for each news article
- The source dataset is stored in `Raw_Dataset.csv`.

---

## Features

- Fetches and parses web content using `BeautifulSoup`
- Removes HTML tags and cleans up article text
- Generates article summaries using the `facebook/bart-large-cnn` transformer
- Extracts image captions based on publisher-specific HTML structure
- Parallel processing for faster performance (via `ThreadPoolExecutor`)
- Exports structured data to `processedData.json`

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Skyzser/Coronavirus-Newspaper-Content-Summarisation-and-Image-Caption-Extraction.git
cd Coronavirus-Newspaper-Content-Summarisation-and-Image-Caption-Extraction
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Add the Dataset
Download `Raw_Dataset.csv` from [this Kaggle notebook](https://www.kaggle.com/code/jwallib/coronavirus-newspaper-classification/notebook) and place it in the project root.

### 4. Run the Script
```bash
python script.py
```

---

## Output

- A JSON file named `processedData.json` is created with structured article metadata, summaries, and captions.

Example entry:
```json
{
        "ID": 1,
        "Metadata": {
            "NewsArticle_Title": "Coronavirus: WHO to decide if deadly virus is international emergency | Metro News",
            "NewsArticle_Publisher": "metro",
            "NewsArticle_Link": "https://metro.co.uk/2020/01/20/world-health-organisation-decide-deadly-coronavirus-international-emergency-12091379/",
            "Datetime_Extracted": "2020-04-06 14:22:27.749206"
        },
        "NewsArticle_Summary": "Crisis talks to be held in Geneva to discuss whether it is an international health emergency. President Xi Jinping said 217 people in China have been infected and called on the government to ramp up monitoring efforts during Chinese New Year. The outbreak is believed to have started from people who picked it up at a seafood market in the city of Wuhan, in central China. A British patient who was \u2018days from death\u2019 is also feared to have been struck with the virus, after showing similar symptoms while on holiday in Thailand. South Korea reported its first case Monday, when a 35-year-old Chinese woman tested positive for the new coronavirus one day after arriving at Seoul\u2019s Incheon airport. The woman has been isolated at a state-run hospital in Incheon city.",
        "NewsArticle_Image_Caption": "It has been confirmed coronavirus spreads by human contact (Picture: EPA/Reuters)"
    },
```

---

## Model Used

- **Summariser**: [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) via Hugging Face Transformers
```
Pros:
  Fine-tuned on the CNN/DailyMail dataset, which is well-suited for news articles.
  Produces coherent and informative summaries.

Cons:
  It can be computationally heavier and slower when processing many rows.
```

---

## Notes

- The script skips articles from *The Times* and *The Telegraph* as they often have paywalls.
- Some URLs may fail to load or parse properly—these are logged and skipped.

---

## License

Open-source under the MIT License.

---
