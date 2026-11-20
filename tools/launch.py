"""Start a long job on the pod so it survives the Jupyter shell that launched it.

`nohup cmd &` from a Jupyter-proxied shell was killed together with that shell several times; starting the
job in a new session is reliable.  Usage on the pod:
    python launch.py /workspace/ad_v2/v2/run_70b.sh /workspace/logs/run_70b.log
"""
import subprocess
import sys

script, log = sys.argv[1], sys.argv[2]
subprocess.Popen(["bash", script], stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                 start_new_session=True, cwd="/workspace")
print("launched", script, "->", log)
