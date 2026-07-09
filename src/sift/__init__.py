from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("sift-search-kde")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
