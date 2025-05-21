import os
import re
import requests
import sys
from urllib.parse import urlparse

def download_arxiv_pdf(url, save_dir="./agent_folder/papers/"):
    """
    Download a PDF from arXiv and save it to the specified directory.
    
    Args:
        url (str): URL to the arXiv PDF
        save_dir (str): Directory to save the PDF
    
    Returns:
        str: Path to the saved PDF
        dict: Metadata extracted from the filename
    """
    # Extract the arXiv ID from the URL
    parsed_url = urlparse(url)
    
    # Handle different URL formats
    if "arxiv.org" in parsed_url.netloc:
        # Extract the arXiv ID using regular expression
        if "/pdf/" in url:
            match = re.search(r'pdf/([^/]+)', url)
            if match:
                arxiv_id = match.group(1)
            else:
                raise ValueError(f"Could not extract arXiv ID from URL: {url}")
        elif "/abs/" in url:
            match = re.search(r'abs/([^/]+)', url)
            if match:
                arxiv_id = match.group(1)
            else:
                raise ValueError(f"Could not extract arXiv ID from URL: {url}")
        else:
            raise ValueError(f"Unsupported arXiv URL format: {url}")
    else:
        raise ValueError(f"Not an arXiv URL: {url}")
    
    # Get the PDF URL
    if "/pdf/" not in url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    else:
        pdf_url = url
    
    # Make sure it ends with .pdf
    if not pdf_url.endswith('.pdf'):
        pdf_url = f"{pdf_url}.pdf" if not pdf_url.endswith('.') else f"{pdf_url}pdf"
    
    # Download the PDF
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the PDF
        filename = f"arxiv_{arxiv_id.replace('.', '_')}.pdf"
        save_path = os.path.join(save_dir, filename)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract metadata (basic info from the ID)
        metadata = {
            "title": f"arXiv:{arxiv_id}",
            "authors": ["Unknown Authors"],
            "year": f"20{arxiv_id.split('.')[0][:2]}" if '.' in arxiv_id else "Unknown Year"
        }
        
        return save_path, metadata
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading PDF: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_arxiv_pdf.py <arxiv_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    save_path, metadata = download_arxiv_pdf(url)
    
    print(f"Downloaded PDF to: {save_path}")
    print(f"Metadata: {metadata}") 