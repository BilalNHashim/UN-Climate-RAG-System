"""
Item pipelines for the climate policy extractor.
"""
import os
import logging

from datetime import datetime, date
from scrapy.pipelines.files import FilesPipeline
from scrapy.exceptions import DropItem
from scrapy import Request
from .models import NDCDocumentModel, init_db, get_db_session
from dotenv import load_dotenv
from itemadapter import ItemAdapter

# # from .logging import setup_colored_logging

logger = logging.getLogger(__name__)

def generate_doc_id(item):
    """Generate a document ID from item metadata."""
    country = item.get('country', 'unknown').lower().replace(" ", "_")
    lang = item.get('language', 'unknown').lower().replace(" ", "_")
    try:
        # Ensure we're using just the date part for the ID
        submission_date = item.get('submission_date')
        if isinstance(submission_date, datetime):
            date_str = submission_date.date().strftime('%Y%m%d')
        elif isinstance(submission_date, date):
            date_str = submission_date.strftime('%Y%m%d')
        else:
            date_str = 'unknown_date'
    except:
        date_str = 'unknown_date'
    
    return f"{country}_{lang}_{date_str}"


class DocumentDownloadPipeline(FilesPipeline):
    """
    Pipeline for downloading NDC documents.
    
    This pipeline:
    1. Downloads PDF documents from URLs
    2. Stores them with appropriate filenames
    3. Updates the item with file paths
    """
    
    def __init__(self, store_uri, download_func=None, settings=None):
        super().__init__(store_uri, download_func, settings)
    
    def get_media_requests(self, item, info):
        """
        Generate requests for downloading files.
        
        Args:
            item: The item containing file URLs
            info: The media pipeline info
            
        Yields:
            Requests for downloading files
        """
        if 'url' in item:
            logger.info(f"Requesting download for: {item['url']}")
            yield Request(
                item['url'],
                meta={'item': item}  # Pass the item in request meta for use in file_path
            )
    
    def file_path(self, request, response=None, info=None, *, item=None):
        """
        Generate file path for storing the document.
        
        This is called after the file has been downloaded and 
        is how the Scrapy documentation recommends doing this.
        
        Args:
            request: The download request
            response: The download response
            info: The media pipeline info
            item: The item being processed
            
        Returns:
            Path where the file should be stored
        """
        # Use item from request.meta if not provided directly
        if item is None:
            item = request.meta.get('item')
            if item is None:
                raise DropItem(f"No item provided in request meta for {request.url}")
        
        # Use item['country'] and item['submission_date'] to compose the filename
        country = item.get('country', 'unknown').lower().replace(" ", "_")
        lang = item.get('language', 'unknown').lower().replace(" ", "_")
        
        try:
            date_str = item.get('submission_date').strftime('%Y%m%d')
        except:
            date_str = 'unknown_date'
        
        # Format the filename to include country and submission date
        file_name = f"{country}_{lang}_{date_str}.pdf"
        
        logger.info(f"Saving file as: {file_name}")
        return file_name
    
    def item_completed(self, results, item, info):
        """
        Called when all file downloads for an item have completed.
        
        Args:
            results: List of (success, file_info_or_error) tuples
            item: The item being processed
            info: The media pipeline info
            
        Returns:
            The updated item
        """
        # Filter out failed downloads
        file_paths = [x['path'] for ok, x in results if ok]
        
        if not file_paths:
            logger.error(f"Failed to download file for {item.get('country', 'unknown')}")
            raise DropItem(f"Failed to download file for {item.get('country', 'unknown')}")
        
        # Update the item with the file path
        item['file_path'] = os.path.join(info.spider.settings.get('FILES_STORE', ''), file_paths[0])
        
        # Get file size
        try:
            item['file_size'] = os.path.getsize(item['file_path'])
        except:
            item['file_size'] = 0
            
        logger.info(f"Successfully downloaded file to: {item['file_path']}")
        return item

class PostgreSQLPipeline:
    """Pipeline for storing NDC documents in PostgreSQL."""

    def __init__(self, db_url=None):
        # Load environment variables
        load_dotenv()
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment variables")

    @classmethod
    def from_crawler(cls, crawler):
        return cls()

    def open_spider(self, spider):
        """Initialize database connection when spider opens."""
        self.logger = spider.logger
        init_db(self.db_url)  # Create tables if they don't exist
        self.session = get_db_session(self.db_url)

    def close_spider(self, spider):
        """Close database connection when spider closes."""
        self.session.close()

    def process_item(self, item, spider):
        """Process scraped item and store in PostgreSQL."""
        adapter = ItemAdapter(item)
        
        # Convert submission_date to date if it's a datetime
        if 'submission_date' in item:
            submission_date = item['submission_date']
            if isinstance(submission_date, datetime):
                item['submission_date'] = submission_date.date()
        
        self.logger.debug(f"Processing item: {item}")

        # Generate doc_id from metadata (same as future file name)
        doc_id = generate_doc_id(item)
        self.logger.debug(f"Generated doc_id: {doc_id}")

        # Create or update document record
        self.logger.debug(f"Querying database for document with doc_id: {doc_id}")
        doc = self.session.query(NDCDocumentModel).filter_by(doc_id=doc_id).first()

        if doc:
            log_msg = (
                f"Document found in database: {doc}. "
                "Checking if it has already been fully processed..."
            )
            self.logger.debug(log_msg)

            retrieved_doc_as_dict = adapter.asdict()
            
            # Check if any data has changed, excluding timestamps we don't want to modify
            has_changes = False
            changes = []
            
            for key, value in retrieved_doc_as_dict.items():
                # Skip downloaded_at and processed_at to preserve their values
                if key in ['downloaded_at', 'processed_at', 'scraped_at']:
                    continue

                if hasattr(doc, key):
                    current_value = getattr(doc, key)
                    
                    if current_value != value:
                        changes.append(f"{key}: {current_value} -> {value}")
                        has_changes = True
                        setattr(doc, key, value)
            
            if has_changes:
                # Always update scraped_at when we see the document
                doc.scraped_at = now_london_time()
                self.logger.info(f"Updating document {doc_id} with changes: {', '.join(changes)}")
            else:
                raise DropItem(f"No changes detected for document {doc_id}, skipping update")
        else:
            log_msg = (
                f"Document not found in database. "
                "Inserting new document."
            )
            self.logger.debug(log_msg)

            doc = NDCDocumentModel(
                doc_id=doc_id,
                country=adapter.get('country'),
                title=adapter.get('title'),
                url=adapter.get('url'),
                language=adapter.get('language'),
                submission_date=adapter.get('submission_date'),
                file_path=None,       # Will be set by download pipeline (outside scrapy)
                file_size=None,       # Will be set by download pipeline (outside scrapy)
                extracted_text=None,  # Will be set by processing pipeline (outside scrapy)
                chunks=None,          # Will be set by processing pipeline (outside scrapy)
                downloaded_at=None,   # Will be set by download pipeline (outside scrapy)
                processed_at=None     # Will be set by processing pipeline (outside scrapy)
            )
            self.logger.debug(f"Adding document to PostgreSQL: {doc}")
            try:
                self.session.add(doc)
            except Exception as e:
                single_line_msg = str(e).replace("\n", " ")
                self.logger.error(f"Error adding document to PostgreSQL: {single_line_msg}")
                raise DropItem(f"Failed to add document to PostgreSQL: {single_line_msg}")
        
        try:
            self.session.commit()
            # Add doc_id back to the item for downstream processing
            item['doc_id'] = doc_id
            self.logger.debug(f"Stored item in PostgreSQL: {item}")
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error storing item in PostgreSQL: {e}")
            raise DropItem(f"Failed to store item in PostgreSQL: {e}")
        
        return item