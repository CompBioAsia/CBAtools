import re

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
        VERSIONFILE = "src/cba_tools/_version.py"
        verstrline = open(VERSIONFILE, "rt").read()
        mo = re.search(VSRE, verstrline, re.M)
        if mo:
            verstr = mo.group(1)
        else:
            raise RuntimeError(f"Unable to find version in {VERSIONFILE}.")

        metadata["version"] = verstr
