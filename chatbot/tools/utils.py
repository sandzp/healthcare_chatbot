import json
import re
import asyncio
from functools import wraps
from datetime import date, datetime


class _CustomEncoder(json.JSONEncoder):
    """JSON encoder that handles date and datetime serialization."""

    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def retry_with_backoff(max_retries: int = 3, exceptions=(Exception,)):
    """Decorator that retries an async function with exponential backoff on transient errors."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    if hasattr(e, "status_code") and e.status_code == 429:
                        match = re.search(r"try again in (\d+\.?\d*)s", str(e))
                        wait = float(match.group(1)) if match else 2 ** attempt
                    else:
                        wait = 2 ** attempt
                    print(f"  [retry] {func.__name__} failed ({type(e).__name__}), attempt {attempt + 1}/{max_retries}, retrying in {wait:.1f}s...", flush=True)
                    await asyncio.sleep(wait)
        return wrapper
    return decorator
