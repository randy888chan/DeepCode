import requests
import os
import re
from urllib.parse import urlparse

def extract_arxiv_id(url_or_id):
    """Extract arXiv ID from a URL or direct ID string."""
    if not url_or_id:
        return None
    
    # Check if it's already just an ID
    if re.match(r'^\d+\.\d+v\d+$', url_or_id) or re.match(r'^\d+\.\d+$', url_or_id):
        return url_or_id
    
    # Try to extract from URL
    parsed_url = urlparse(url_or_id)
    path = parsed_url.path
    
    # Extract ID from path
    match = re.search(r'((?:\d{4}\.\d{4,5})|(?:\d{7}))(?:v\d+)?', path)
    if match:
        return match.group(0)
    
    return None

def download_arxiv_pdf(arxiv_id, save_dir):
    """Download a paper from arXiv given its ID."""
    if not arxiv_id:
        return {"status": "failure", "message": "Invalid arXiv ID"}
    
    # Clean the arXiv ID
    arxiv_id = extract_arxiv_id(arxiv_id)
    if not arxiv_id:
        return {"status": "failure", "message": "Could not extract valid arXiv ID"}
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        # Download the PDF
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        # Save the PDF
        file_path = os.path.join(save_dir, f"{arxiv_id}.pdf")
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Get metadata from arXiv API
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        metadata_response = requests.get(api_url)
        metadata_response.raise_for_status()
        
        # Extract basic metadata (simple parsing, not full XML parsing)
        content = metadata_response.text
        title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Unknown"
        if "Error" in title or "arXiv" in title:
            # Skip the first title (which is usually "Error" or "arXiv Query")
            title_matches = re.findall(r'<title>(.*?)</title>', content, re.DOTALL)
            title = title_matches[1].strip() if len(title_matches) > 1 else "Unknown"
        
        # Extract authors
        author_matches = re.findall(r'<name>(.*?)</name>', content)
        authors = [author.strip() for author in author_matches]
        
        # Extract year
        year_match = re.search(r'<published>(.*?)</published>', content)
        year = year_match.group(1)[:4] if year_match else "Unknown"
        
        return {
            "status": "success",
            "paper_path": file_path,
            "metadata": {
                "title": title,
                "authors": authors,
                "year": year
            }
        }
    except Exception as e:
        return {"status": "failure", "message": str(e)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        arxiv_id = sys.argv[1]
        save_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        result = download_arxiv_pdf(arxiv_id, save_dir)
        print(result)
    else:
        print("Usage: python download_arxiv_pdf.py <arxiv_id> [save_directory]")