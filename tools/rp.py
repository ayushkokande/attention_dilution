import json, os, sys, requests
KEY = open(os.path.expanduser("~/.secrets/runpod_key")).read().strip()
def gql(q, v=None):
    r = requests.post("https://api.runpod.io/graphql", json={"query": q, "variables": v or {}},
                      headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
    d = r.json()
    if "errors" in d: raise SystemExit(json.dumps(d["errors"])[:800])
    return d["data"]
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "create":
        pw = open(os.path.expanduser("~/.secrets/jupyter_pw")).read().strip()
        hf = open(os.path.expanduser("~/.secrets/hf_token")).read().strip()
        inp = dict(cloudType="SECURE", gpuCount=2, gpuTypeId=sys.argv[2] if len(sys.argv) > 2 else "NVIDIA A100-SXM4-80GB",
                   name=os.environ.get("POD_NAME","attention-dilution-v2"), imageName="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                   containerDiskInGb=60, volumeInGb=200, volumeMountPath="/workspace", ports="8888/http,22/tcp",
                   env=[{"key": "JUPYTER_PASSWORD", "value": pw}, {"key": "HF_TOKEN", "value": hf}])
        d = gql("mutation($i: PodFindAndDeployOnDemandInput){ podFindAndDeployOnDemand(input:$i){ id desiredStatus machineId } }", {"i": inp})
        print(json.dumps(d))
    elif cmd == "status":
        d = gql("query($i: PodFilter){ pod(input:$i){ id desiredStatus runtime{ uptimeInSeconds ports{ ip isIpPublic privatePort publicPort type } gpus{ id gpuUtilPercent memoryUtilPercent } } } }", {"i": {"podId": sys.argv[2]}})
        print(json.dumps(d))
    elif cmd == "stop":
        print(json.dumps(gql("mutation($i: PodStopInput!){ podStop(input:$i){ id desiredStatus } }", {"i": {"podId": sys.argv[2]}})))
    elif cmd == "list":
        print(json.dumps(gql("query { myself { clientBalance pods { id name desiredStatus } } }")))
