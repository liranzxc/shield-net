# Define the shield_net decorator
import functools
from typing import Callable

from fastapi import HTTPException


def shield_net(func: Callable):
    """
    A decorator to add pre-processing, post-processing, or error handling
    around the wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Pre-processing (e.g., logging, input validation)
            print(f"shield_net: Called function {func.__name__} with args={args}, kwargs={kwargs}")

            # Call the original function
            result = func(*args, **kwargs)

            # Post-processing (e.g., additional logging)
            print(f"shield_net: Function {func.__name__} returned {result}")
            return result
        except Exception as e:
            # Error handling
            print(f"shield_net: An error occurred in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return wrapper
