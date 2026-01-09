# Simple in-memory cache dictionary
cache = {}


def get_from_cache(key: str):
    """
    Check if a key exists in the cache.
    Returns the cached value or None if not present.
    """
    return cache.get(key)


def add_to_cache(key: str, value: dict):
    """
    Add a value to the cache with the given key.
    """
    cache[key] = value
