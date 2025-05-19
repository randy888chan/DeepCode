import requests
import os
from pathlib import Path
import time

def get_next_paper_number(papers_dir):
    """
    Get the next available paper number based on existing files
    
    Args:
        papers_dir: Directory containing paper files
    Returns:
        int: Next available paper number
    """
    if not os.path.exists(papers_dir):
        return 1
        
    existing_files = [f for f in os.listdir(papers_dir) if f.startswith('paper_') and f.endswith('.pdf')]
    if not existing_files:
        return 1
        
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1

def download_arxiv_pdf(arxiv_id, output_dir="./Deepcode/papers"):
    """
    Download a PDF file from arXiv and save it with sequential numbering
    
    Args:
        arxiv_id: arXiv paper ID (e.g., '2403.00813')
        output_dir: Directory to save the PDF file (default: './Deepcode/papers')
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get next available paper number
    paper_number = get_next_paper_number(output_dir)
    output_filename = f"paper_{paper_number:02d}.pdf"
    output_path = output_dir / output_filename
    
    # Format arXiv ID (remove version number if present)
    arxiv_id = arxiv_id.split('v')[0]
    
    # Create a session to handle cookies and redirects
    session = requests.Session()
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,application/x-pdf',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://arxiv.org/'
    }
    
    try:
        # First, visit the abstract page to get any necessary cookies
        abstract_url = f"https://arxiv.org/abs/{arxiv_id}"
        print(f"Visiting abstract page: {abstract_url}")
        session.get(abstract_url, headers=headers, timeout=10)
        
        # Now try to download the PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"\nAttempting to download PDF from: {pdf_url}")
        print(f"Will save to: {output_path}")
        
        # Make the request with the session
        response = session.get(
            pdf_url,
            headers=headers,
            stream=True,
            timeout=30,
            allow_redirects=True
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Final URL after redirects: {response.url}")
        
        if response.status_code == 200:
            # Check if the response is actually a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            print(f"Content-Type: {content_type}")
            
            if 'application/pdf' in content_type or 'application/x-pdf' in content_type:
                # Get the total file size
                total_size = int(response.headers.get('Content-Length', 0))
                print(f"Total file size: {total_size / 1024:.2f} KB")
                
                # Download the file with progress tracking
                downloaded_size = 0
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            # Print progress
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\rDownload progress: {progress:.1f}%", end='')
                
                print("\nDownload completed!")
                print(f"File saved to: {output_path}")
                print(f"Final file size: {downloaded_size / 1024:.2f} KB")
                
                # Verify the file exists and has content
                if output_path.exists() and output_path.stat().st_size > 0:
                    print("File verification successful!")
                    return True
                else:
                    print("Error: File was not saved properly!")
                    return False
            else:
                print(f"Error: Response is not a PDF. Content-Type: {content_type}")
                print(f"Response content preview: {response.text[:200]}")
        else:
            print(f"Error: Failed to download PDF. Status code: {response.status_code}")
            print(f"Response content preview: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Please check your internet connection.")
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Please check your internet connection.")
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        print(f"Error type: {type(e).__name__}")
    
    return False

if __name__ == "__main__":
    # Example usage
    arxiv_id = "2403.00813"  # The arXiv ID from your URL
    
    try:
        success = download_arxiv_pdf(arxiv_id)
        if not success:
            print("Failed to download the PDF. Please check the error messages above.")
    except Exception as e:
        print(f"Error: {str(e)}") 