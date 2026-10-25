"""Console output, optionally mirrored to a file."""


class Log:
    """Console, mirrored to a file when one is configured."""

    def __init__(self, path=None, echo=True):
        self.echo = echo
        self.fh = self._open(path)

    @staticmethod
    def _open(path):
        return open(path, "a", encoding="utf-8") if path else None

    def __call__(self, text=""):
        self._write(text + "\n")

    def raw(self, text):
        """A streamed chunk: no newline, but flushed so it shows up at once."""
        self._write(text)

    def _write(self, text):
        if self.echo:
            print(text, end="", flush=True)
        if self.fh:
            self.fh.write(text)
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


log = Log()


def setup(path=None, echo=True):
    """Point the shared sink at a file and/or the console."""
    log.echo = echo
    if path and not log.fh:
        log.fh = log._open(path)
