import functools
from typing import Callable

from fastapi import HTTPException

from api.shield_net.services import is_safe_prompt, post_process_content


def shield_net(func: Callable):
    """
    A decorator to add pre-processing, post-processing, or error handling
    around the wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Pre-processing (e.g., logging, input validation)
            print(f"inside shield_net decorator")
            system_prompt = kwargs["system_prompt"]
            prompt = kwargs["prompt"]

            if not (is_safe_prompt(system_prompt) and is_safe_prompt(prompt)):
                return "ShieldNet Model Detection - Not Safe"

            # Call the original function
            result = func(*args, **kwargs)
            post_process_result = post_process_content(result)
            # Post-processing (e.g., additional logging)
            # print(f"shield_net: Function {func.__name__} returned {post_process_result}")
            return post_process_result
        except Exception as e:
            # Error handling
            print(f"shield_net: An error occurred in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return wrapper
