import os
import tempfile

# Tests never touch the network: every external lookup is reported as unavailable,
# unless a test injects its own fake HTTP client.
os.environ["INTEL_MODE"] = "off"
os.environ.setdefault("DEMO_INBOX_DIR", os.path.join(tempfile.mkdtemp(), "live"))
