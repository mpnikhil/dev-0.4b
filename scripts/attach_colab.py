"""Adopt an existing user-designated runtime through the installed CLI's state API.

Run with the Python interpreter belonging to google-colab-cli. No new VM is allocated.
Never prints or exports authentication tokens.
"""
import argparse
from pathlib import Path
from colab_cli.common import state
from colab_cli.state import SessionState

p = argparse.ArgumentParser()
p.add_argument("--endpoint", required=True)
p.add_argument("--name", default="jev-pilot")
p.add_argument("--config", default=".cache/colab-sessions.json")
a = p.parse_args()
state.config_path = str(Path(a.config).resolve())
matches = [r for r in state.client.list_assignments() if r.endpoint == a.endpoint]
if len(matches) != 1:
    raise SystemExit("Designated runtime no longer exists; no changes made")
r = matches[0]
state.store.add(SessionState(name=a.name, endpoint=r.endpoint, token=r.runtime_proxy_info.token,
                             url=r.runtime_proxy_info.url, variant=r.variant.name,
                             accelerator=r.accelerator.value))
Path(state.config_path).chmod(0o600)
print(f"Attached existing {r.accelerator.value} runtime as {a.name}; no VM allocated")
