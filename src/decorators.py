# decorators.py

import time

def time_function(func):
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Args:
        func: Fonction à chronométrer
    
    Returns:
        Fonction décorée
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Fonction {func.__name__} exécutée en {elapsed_time:.2f} secondes")
        return result, elapsed_time
    return wrapper