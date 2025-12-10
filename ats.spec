# ats.spec – final working macOS version
import os
import glob
from PyInstaller.utils.hooks import collect_submodules

# PyInstaller defines `specnm` — the absolute path to this .spec file
# It is the full absolute path of the .spec file
root = os.path.dirname(os.path.abspath(specnm))

print("ATS SPEC ROOT:", root)  # optional debug print


packages = ["audio", "core", "gui", "image", "extras"]
search_paths = [os.path.join(root, p) for p in packages]

# AUTO-COLLECT ALL PYTHON FILES
datas = []
for p in packages:
    folder = os.path.join(root, p)
    if os.path.isdir(folder):
        for file in glob.glob(folder + "/**/*", recursive=True):
            if os.path.isfile(file) and not file.endswith(".pyc"):
                rel = os.path.relpath(file, root)
                datas.append((file, os.path.dirname(rel)))

# OPTIONAL ICON
icon_file = os.path.join(root, "icon.icns")
icon_arg = icon_file if os.path.exists(icon_file) else None

# IMPORT HOOKS
hiddenimports = []
hiddenimports += collect_submodules("simpleaudio")
hiddenimports += collect_submodules("matplotlib")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("scipy")

# EXECUTABLE
a = Analysis(
    ["main.py"],
    pathex=[root],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    binaries=[],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="ATS",
    icon=icon_arg,
    console=False,
)

app = BUNDLE(
    exe,
    name="ATS.app",
    icon=icon_arg,
    bundle_identifier="com.austin.ats",
)
