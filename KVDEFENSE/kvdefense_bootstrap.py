"""Site-processing bootstrap for KVDEFENSE, invoked from a .pth file in
venv_dynamo_pd's site-packages (001_kvdefense.pth), following the same
pattern as KVHOOK's 000_kvhook.pth/kvhook_bootstrap.py (see that file for
why .pth "import" lines are used instead of sitecustomize.py on this host).

Inert unless KVDEFENSE_ENABLE=1. Never raises -- a bug here must not be able
to break normal use of this pinned venv, including ordinary (undefended)
serving runs where this variable is simply unset.
"""

import os


def _activate():
    if os.environ.get("KVDEFENSE_ENABLE") != "1":
        return
    try:
        import kvdefense_patch

        kvdefense_patch.patch_block_pool()
    except Exception:
        import traceback

        try:
            log_dir = os.path.expanduser("~/LG2026/KVDEFENSE")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "kvdefense_patch.log"), "a") as f:
                f.write("[KVDEFENSE bootstrap] patch_block_pool() failed:\n")
                f.write(traceback.format_exc() + "\n")
        except OSError:
            pass


_activate()
