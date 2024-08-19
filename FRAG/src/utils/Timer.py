import time
from functools import wraps


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs) 
        end_time = time.time() 
        execution_time = end_time - start_time 
        print(
            f"Function '{func.__qualname__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper


class TimedClass:
    def __init__(self):
        for attr_name in dir(self):
            if attr_name == "process":
                attr_value = getattr(self, attr_name)
                if callable(attr_value):
                    setattr(self, attr_name, timer_decorator(attr_value))
