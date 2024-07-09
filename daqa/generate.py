import argparse
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time

import datasets
import mwparserfromhell
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_article(title, text):
    logging.debug(f"Processing article: {title}")
    if is_redirect(text):
        logging.info(f"Article {title} is a redirect, skipping")
        return None
    try:
        wikicode = mwparserfromhell.parse(text)
    except Exception as e:
        logging.error(f"Error parsing wikitext for article {title}: {e}")
        return {"title": title, "content": text}

    content = clean_wikitext(str(wikicode))

    if is_include_only(content):
        logging.info(f"Article {title} is an include only or primarily navbox, skipping")
        return None
    
    if not is_meaningful_article(content):
        logging.info(f"Article {title} is not meaningful, skipping")
        return None

    logging.debug(f"Processed article: {title}")
    return {"title": title, "content": content}

def is_meaningful_article(content):
    min_content_length = 300
    min_word_count = 50
    
    word_count = len(content.split())
    
    return len(content) >= min_content_length and word_count >= min_word_count

def is_include_only(content):
    # Remove any leading whitespace and newlines
    content = content.lstrip()

    # Check if the content starts with common 'include only' patterns
    include_only_starts = [
        '}',  # Closing bracket of a template
        '|',  # Parameter separator in a template
        '</'  # Closing tag
    ]

    if any(content.startswith(start) for start in include_only_starts):
        return True

    # Check if the content consists only of templates, categories, and whitespace
    cleaned_content = re.sub(r'{{[^}]*}}|\[\[Kategori:[^\]]*\]\]|\s', '', content)
    if not cleaned_content:
        return True

    # Check for specific 'include only' templates
    include_only_templates = [
        r'{{\s*(?:Include only|Kun til inklusion)',
        r'{{\s*(?:navbox|infoboks|Infoboks)',
        r'{{\s*(?:tabel|Table)',
    ]

    for template in include_only_templates:
        if re.search(template, content, re.IGNORECASE):
            # Check if this template is the main content of the article
            template_ratio = len(re.findall(template, content, re.IGNORECASE)) / len(content.split())
            if template_ratio > 0.8:  # If more than 80% of the words are part of these templates
                return True

    return False


def generate_questions(article, cache_dir):
    logging.debug(f"Generating questions for article: {article['title']}")
    article_hash = hashlib.md5((article['title'] + article['content']).encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{article_hash}.json")

    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            logging.debug(f"Loading cached questions for article: {article['title']}")
            return json.loads(f.read())

    prompt = f"""
    Givet følgende Wikipedia-artikel, skal du generere 5 spørgsmål og deres svar baseret på indholdet. 
    Svaret skal stå direkte i den givne artikel, uden brug af anden viden. Svaret skal være kort, gerne kun 1 til 2 ord eller et tal.
    Spørgsmålet må gerne være svært, og omformuler gerne ord i teksten.
    Formater outputtet som en liste af `dict`s, hvor hver `dict` indeholder en 'spørgsmål' og en 'svar' nøgle.

    Titel: {article['title']}
    Indhold: {article['content'][:6000]}

    Output kun listen af `dict`s, uden yderligere tekst. Sørg for at både spørgsmål og svar er på dansk.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logging.debug(f"Sending API request for article: {article['title']} (Attempt {attempt + 1})")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates questions and answers based on given content in Danish."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            if response.choices[0].message.content is None:
                raise ValueError("Empty response from API")

            qa_pairs = json.loads(response.choices[0].message.content)

            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(qa_pairs, ensure_ascii=False, indent=2))

            logging.debug(f"Generated and cached questions for article: {article['title']}")
            return qa_pairs
        except Exception as e:
            logging.warning(f"Error occurred while generating Q&A pairs: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    logging.error(f"Failed to generate Q&A pairs after {max_retries} attempts for article: {article['title']}")
    return []

def clean_wikitext(text):
    text = re.sub(r'{{[^}]*}}', '', text)
    text = re.sub(r'\[\[Kategori:[^\]]*\]\]', '', text)
    text = re.sub(r'\[http[^\]]*\]', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    return text.strip()

def is_redirect(text):
    return text.strip().lower().startswith('#redirect')

def process_articles(db_path, article_ids, cache_dir):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    dataset = datasets.Dataset.from_dict({"title": [], "context": [], "question": [], "answer": []})

    with tqdm(total=len(article_ids), desc="Processing articles") as pbar:
        for article_id in article_ids:
            cursor.execute('SELECT title, content FROM articles WHERE id = ?', (article_id,))
            result = cursor.fetchone()
            if result:
                title, text = result
                article = process_article(title, text)
                if article:
                    qa_pairs = generate_questions(article, cache_dir)
                    for qa in qa_pairs:
                        qa_entry = {
                            "title": title,
                            "context": article["content"],
                            "question": qa["spørgsmål"],
                            "answer": qa["svar"]
                        }
                        dataset = dataset.add_item(qa_entry)
                pbar.update(1)

    conn.close()
    return dataset

def main(args):
    db_path = "danish_wikipedia.db"
    cache_dir = "qa_cache"
    os.makedirs(cache_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM articles')
    total_articles = cursor.fetchone()[0]
    
    article_count = args.limit if args.limit else total_articles
    article_ids = list(range(1, total_articles + 1))
    random.seed(args.seed)
    random.shuffle(article_ids)
    selected_article_ids = article_ids[:article_count]

    conn.close()

    try:
        dataset = process_articles(db_path, selected_article_ids, cache_dir)

        dataset.save_to_disk("daqa")

        if args.upload:
            api = HfApi()
            repo_id = args.repo_id
            repo = api.create_repo(repo_id, private=True, repo_type="dataset", exist_ok=False)
            dataset.push_to_hub(repo_id)
            logging.info(f"Dataset uploaded to Hugging Face Hub: {repo_id}")
        else:
            logging.info(f"Q&A dataset saved to {dataset}")
    except Exception as e:
        logging.error(f"Error in main processing loop: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q&A dataset from Danish Wikipedia")
    parser.add_argument("--limit", type=int, help="Limit the number of articles to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling articles")
    parser.add_argument("--upload", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, help="Hugging Face Hub repository ID for uploading")
    args = parser.parse_args()

    if args.upload and not args.repo_id:
        parser.error("--repo-id is required when --upload is set")

    try:
        main(args)
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}", exc_info=True)