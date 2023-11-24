from contextlib import contextmanager


class ServerException(Exception):
    def __init__(self, e: Exception):
        self.reason = str(e)


@contextmanager
def server_exception_contextmanager():
    try:
        yield
    except Exception as e:
        raise ServerException(e)
