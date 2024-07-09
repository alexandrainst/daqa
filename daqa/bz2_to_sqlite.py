import bz2
import sqlite3
import xml.etree.ElementTree as ET

from tqdm import tqdm


def preprocess_wikipedia_dump(bz2_file_path, db_file_path):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Create table for articles
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY,
        title TEXT,
        content TEXT
    )
    ''')

    # Define the XML namespace
    ns = {'wiki': 'http://www.mediawiki.org/xml/export-0.11/'}

    with bz2.open(bz2_file_path, 'rt', encoding='utf-8') as file:
        context = ET.iterparse(file, events=('end',))
        
        # Use tqdm to show progress
        with tqdm(desc="Processing articles") as pbar:
            for event, elem in context:
                if elem.tag.endswith('page'):
                    title_elem = elem.find('.//wiki:title', namespaces=ns)
                    text_elem = elem.find('.//wiki:text', namespaces=ns)
                    
                    if title_elem is not None and text_elem is not None:
                        title = title_elem.text
                        content = text_elem.text
                        
                        if title and content:
                            cursor.execute('INSERT INTO articles (title, content) VALUES (?, ?)', (title, content))
                            conn.commit()
                            pbar.update(1)
                    
                    # Clear the element to free up memory
                    elem.clear()

    conn.close()

if __name__ == '__main__':
    bz2_file_path = 'dawiki-latest-pages-articles.xml.bz2'
    db_file_path = 'danish_wikipedia.db'
    preprocess_wikipedia_dump(bz2_file_path, db_file_path)