# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "build-plan.json").read_text())
stage = "/home/enntitysparkadmin/glm-fp4-groups-20261020-build"


def ssh(args, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15",
                           "ennspark01", shlex.join(args)], **kwargs)


assert ssh(["docker", "image", "inspect", "--format", "{{.Id}}", "localhost/atlas-sparkglm:native-mla-fuse-a7ac2c046d78"],check=True,capture_output=True,text=True).stdout.strip()==plan['base_image']
ssh(["mkdir", stage], check=True)
with (ROOT / "context.tar").open("rb") as stream:
    ssh(["tar", "-xf", "-", "-C", stage], stdin=stream, check=True)
with (ROOT / "build.log").open("wb") as log:
    result = ssh(["docker", "build", "--pull=false", "--progress=plain",
                  "-t", plan["image"], stage], stdout=log, stderr=subprocess.STDOUT)
(ROOT / "build-exit.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
assert result.returncode == 0, "RC build failed; see build.log"
image = ssh(["docker", "image", "inspect", "--format", "{{.Id}}", plan["image"]],
            check=True, capture_output=True, text=True).stdout.strip()
code = ("import json,pathlib,hashlib;print(json.dumps({"
        "'binary_sha256':hashlib.sha256(pathlib.Path('/usr/local/bin/spark').read_bytes()).hexdigest(),"
        "'source':json.loads(pathlib.Path('/opt/atlas/source-manifest.json').read_text())}))")
identity = json.loads(ssh(["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", code],
                         check=True, capture_output=True, text=True).stdout)
assert identity["source"]["commit"] == plan["commit"]
identity.update(image_id=image, tag=plan["image"], install_tree="experimental")
(ROOT / "candidate-image.json").write_text(json.dumps(identity, indent=2) + "\n")
pipe = "set -o pipefail; docker save " + shlex.quote(plan["image"]) + " | ssh -o BatchMode=yes ennspark02-fab docker load"
ssh(["bash", "-c", pipe], check=True, stdout=subprocess.DEVNULL)
other = ssh(["ssh", "-o", "BatchMode=yes", "ennspark02-fab", "docker", "image", "inspect",
             "--format", "{{.Id}}", plan["image"]], check=True, capture_output=True, text=True).stdout.strip()
assert other == image
(ROOT / "transfer.json").write_text(json.dumps({"both_ranks_image": image}) + "\n")
print("RC built and verified on both ranks:", image, flush=True)
