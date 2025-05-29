import requests
import logging
from bs4 import BeautifulSoup
import pypdf
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentFetcher:
    """Fetches and processes content from URLs"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def fetch_content(self, url: str) -> str:
        """Fetch and extract text content from URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            if "application/pdf" in content_type:
                return self._extract_pdf_text(response.content)
            else:
                return self._extract_html_text(response.text)

        except Exception as e:
            return ""

    def _extract_html_text(self, html: str) -> str:
        """Extract clean text from HTML"""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract text
            text = soup.get_text()

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n")]
            cleaned_text = "\n".join([line for line in lines if line])

            return cleaned_text[:10000]  # Limit to 10k characters
        except Exception as e:
            logger.error(f"Failed to extract HTML text: {e}")
            return ""

    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = pypdf.PdfReader(pdf_file)

            text = ""
            # Extract from first 5 pages only to avoid memory issues
            num_pages = min(len(pdf_reader.pages), 5)
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                text += page.extract_text()

            return text[:10000]  # Limit to 10k characters
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            return ""
