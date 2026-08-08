"""
Site-processing bootstrap for KVHOOK, invoked from a .pth file in
venv_dynamo_pd's site-packages (000_kvhook.pth) rather than
sitecustomize.py.

Why: `import sitecustomize` resolves to /usr/lib/python3.12/sitecustomize.py
(the system stdlib copy) ahead of the venv's own site-packages copy on this
host -- confirmed empirically (KVHOOK_ENABLE=1 python3 -c "import
sitecustomize; print(sitecustomize.__file__)" printed the /usr/lib path,
not the venv one). .pth "import" lines are executed directly during
site.addsitedir() for the directory they live in, so they aren't subject
to that same module-name shadowing.

Inert unless KVHOOK_ENABLE=1. Never raises -- a bug here must not be able
to break normal use of this pinned venv.
"""

import os


def _activate():
    if os.environ.get("KVHOOK_ENABLE") != "1":
        return
    try:
        import kvhook_patch_0251

        kvhook_patch_0251.patch_connector()
    except Exception:
        import traceback

        try:
            log_dir = os.path.expanduser("~/LG2026/KVHOOK/dumps")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "kvhook_patch.log"), "a") as f:
                f.write("[KVHOOK bootstrap] patch_connector() failed:\n")
                f.write(traceback.format_exc() + "\n")
        except OSError:
            pass


_activate()
