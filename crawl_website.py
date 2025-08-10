import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os

def save_texts_to_file(texts, filename, format='json'):
    """
    Save crawled texts to a file in different formats
    
    Args:
        texts: List of text strings
        filename: Output filename
        format: 'json', 'txt', or 'jsonl' (JSON Lines)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    if format == 'json':
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)
    elif format == 'txt':
        with open(filename, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"=== Page {i+1} ===\n")
                f.write(text)
                f.write("\n\n" + "="*50 + "\n\n")
    elif format == 'jsonl':
        with open(filename, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                json.dump({"page_number": i+1, "content": text}, f, ensure_ascii=False)
                f.write('\n')
    
    print(f"Saved {len(texts)} pages to {filename}")

def crawl_bookshelf(start_url, max_pages=100):
    visited = set()
    to_visit = [start_url]
    texts = []
    
    print(f"Starting crawl from: {start_url}")
    print(f"Maximum pages to crawl: {max_pages}")
    print("-" * 60)

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        
        # Show progress
        progress = f"[{len(visited)}/{max_pages}]"
        print(f"{progress} Visiting: {url}")
        
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Extract main content (customize selector as needed)
            main_text = soup.get_text(separator="\n")
            texts.append(main_text)
            
            # Count new links found
            new_links_count = 0
            # Find links to inner pages
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                
                # Check if link is HTML/HTM (skip PDFs, images, etc.)
                parsed_link = urlparse(link)
                path = parsed_link.path.lower()
                
                # Skip non-HTML files
                if path.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                                '.zip', '.rar', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', 
                                '.bmp', '.svg', '.mp3', '.mp4', '.avi', '.mov', '.css', '.js')):
                    continue
                
                # Only include HTML files or pages without extensions (assuming they're HTML)
                if not (path.endswith(('.html', '.htm')) or '.' not in path.split('/')[-1]):
                    continue
                
                # Only crawl links within the same domain
                if urlparse(link).netloc == urlparse(start_url).netloc and link not in visited and link not in to_visit:
                    to_visit.append(link)
                    new_links_count += 1
            
            print(f"    ✓ Content extracted ({len(main_text)} chars), found {new_links_count} new links")
            
        except Exception as e:
            print(f"    ✗ Failed to fetch {url}: {e}")
    
    print("-" * 60)
    print(f"Crawling completed! Visited {len(visited)} pages, extracted {len(texts)} documents")
    return texts

# Example usage:
all_texts = crawl_bookshelf("https://bookshelf.erwin.com/bookshelf/15.0DIBookshelf/Content/Home.htm")

# Save the crawled texts to files
if all_texts:
    # Save as JSON (structured format)
    save_texts_to_file(all_texts, "datasets/website/crawled_content.json", format='json')
    
    # Save as plain text (readable format)
    save_texts_to_file(all_texts, "datasets/website/crawled_content.txt", format='txt')
    
    # Save as JSON Lines (good for processing line by line)
    save_texts_to_file(all_texts, "datasets/website/crawled_content.jsonl", format='jsonl')
    
    print(f"Successfully crawled and saved {len(all_texts)} pages")
else:
    print("No content was crawled")
save_texts_to_file(all_texts, "datasets/website/crawled_texts.json", format="json")
