"""Utility functions for the climate policy extractor."""

from datetime import datetime
import zoneinfo

def now_london_time():
    """Get current time in London timezone.
    
    Returns:
        datetime: Current time in London timezone
    """
    london_tz = zoneinfo.ZoneInfo('Europe/London')
    return datetime.now(london_tz) 