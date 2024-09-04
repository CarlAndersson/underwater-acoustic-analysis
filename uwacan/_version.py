version = None
# This version will be overwritten by versioningit during build,
# to a static version string.


def git_version(root):
    """Get the package version from git tags."""
    import subprocess
    import re

    cmd = ["git", "describe", "--tags", "--dirty", "--always"]
    try:
        p_out = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        return None
    # Format the version
    # We go with <tag>.post<distance>+g<rev>.<dirty>
    description = p_out.stdout.decode().strip().lstrip("Vversion")
    match = re.match(
        r"(?P<tag>[a-zA-Z0-9.]+)(-(?P<distance>[0-9]+))?(-g(?P<rev>[0-9a-f]+))?(?P<dirty>-dirty)?", description
    )
    if not match:
        return None
    version = match["tag"]
    if match["distance"]:
        version += f".post{match['distance']}"
    if match["rev"]:
        version += f"+g{match['rev']}"
    if match["dirty"]:
        if match["rev"]:
            version += "-dirty"
        else:
            version += "+dirty"
    return version


# If there is no hardcoded version string,
# we get one from git.
if version is None:
    import os.path

    version = git_version(os.path.dirname(__file__))
