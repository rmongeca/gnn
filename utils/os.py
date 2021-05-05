from pathlib import Path


def empty_dirs(dirs=None):
    if dirs is None:
        return
    elif isinstance(dirs, (Path, str)):
        dirs = [Path(dirs)]
    for _dir in dirs:
        assert isinstance(_dir, Path)
        for file in [f for f in _dir.glob("*") if f.is_file()]:
            file.unlink()
