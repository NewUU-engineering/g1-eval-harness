import os, sys
if sys.platform.startswith("linux"):          # EGL only exists on Linux
    os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
