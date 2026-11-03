MODE = 0

class inferenceMode:
    def __init__(self):
        pass

    def __enter__(self):
        global MODE
        MODE = 1
        return MODE

    def __exit__(self, exc_type, exc_val, exc_tb):
        global MODE
        MODE = 0

