import subprocess


def get_git_revision_short_hash() -> str:
    """
    Returns current git hash

    Returns: short git hash
    """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
