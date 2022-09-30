#! /usr/bin/env python3

import os
import re
import subprocess
from pathlib import Path
from typing import Dict

import requests
from skopt import gp_minimize, load
from skopt.callbacks import CheckpointSaver
from skopt.plots import plot_convergence
from skopt.utils import use_named_args

from space import ALLOWED_SERVER_SETTINGS, SPACE

# From https://console.dev.neon.tech/app/settings/api-keys
NEON_API_KEY = os.environ["NEON_API_KEY"]
NEON_API_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {NEON_API_KEY}",
}


def create_project(settings: Dict):
    data = {
        "project": {
            "name": f"Neon Hackaton (autocreated)",
            "platform_id": "aws",
            "region_id": "eu-west-1",
            "settings": settings,
        }
    }

    response = requests.post(
        url="https://console.dev.neon.tech/api/v1/projects",
        json=data,
        headers=NEON_API_HEADERS,
    )
    if not response.ok:
        raise RuntimeError(f"Can't create project with {settings=}, {response.text}")

    project = response.json()
    return (
        project["id"],
        [
            f"{role['dsn']}/main"
            for role in project["roles"]
            if role["name"] != "web_access"
        ][0],
    )


def delete_project(project_id):
    response = requests.post(
        url=f"https://console.dev.neon.tech/api/v1/projects/{project_id}/delete",
        headers=NEON_API_HEADERS,
    )

    if not response.ok:
        raise RuntimeError(f"Can't delete project with {project_id=}, {response.text}")

    return response.json()


def parse_pgbench_initialize_output(output: str) -> Dict[str, float]:
    regex = re.compile(
        r"done in (\d+\.\d+) s "
        r"\("
        r"(?:drop tables (\d+\.\d+) s)?(?:, )?"
        r"(?:create tables (\d+\.\d+) s)?(?:, )?"
        r"(?:server-side generate (\d+\.\d+) s)?(?:, )?"
        r"(?:vacuum (\d+\.\d+) s)?(?:, )?"
        r"(?:primary keys (\d+\.\d+) s)?(?:, )?"
        r"\)\."
    )

    last_line = output.splitlines()[-1]

    if (m := regex.match(last_line)) is not None:
        (
            total,
            drop_tables,
            create_tables,
            server_side_generate,
            vacuum,
            primary_keys,
        ) = [float(v) for v in m.groups() if v is not None]
    else:
        print(output)
        raise RuntimeError("Something went wrong")

    return {
        "total": total,
        "drop_tables": drop_tables,
        "create_tables": create_tables,
        "server_side_generate": server_side_generate,
        "vacuum": vacuum,
        "primary_keys": primary_keys,
    }


DIMENTIONS = [s[0] for s in SPACE]
X0 = [s[1] for s in SPACE]
TIMEOUT = 200

@use_named_args(DIMENTIONS)
def pgbench(**options):
    server_settings = {}
    client_settings = {}
    for name, value in options.items():
        value = str(value)
        if name in ALLOWED_SERVER_SETTINGS:
            server_settings[name] = value
        else:
            client_settings[name] = value

    project_id, dsn = create_project(server_settings)

    try:
        stderr = Path("stderr.txt")
        stdout = Path("stdout.txt")

        set_command = ""
        for k, v in client_settings.items():
            set_command += f"ALTER DATABASE main SET {k}={v};"
        subprocess.run(
            ["/opt/homebrew/bin/psql", dsn, "-c", set_command], timeout=TIMEOUT, check=True
        )

        with (stderr.open("w") as err, stdout.open("w") as out):
            subprocess.run(
                [
                    "/opt/homebrew/bin/pgbench",
                    dsn,
                    "--initialize",
                    "--init-steps",
                    "dtGvp",
                    "--scale",
                    "100",
                ],
                stdout=out,
                stderr=err,
                text=True,
                timeout=TIMEOUT,
                check=True,
            )
        timings = parse_pgbench_initialize_output(stderr.read_text())
        return timings["total"]
    except Exception as exc:
        print(exc)
        return TIMEOUT
    finally:
        delete_project(project_id)
        stderr.unlink(missing_ok=True)
        stdout.unlink(missing_ok=True)


def main():
    checkpoint = Path("./checkpoint.pkl")
    if checkpoint.exists():
        print("*** Loading checkpoint ***")
        res = load(checkpoint)
        x0 = res.x_iters
        y0 = res.func_vals
    else:
        x0 = X0
        y0 = None

    checkpoint_saver = CheckpointSaver(checkpoint)
    result = gp_minimize(
        pgbench,
        DIMENTIONS,
        n_calls=1000,
        x0=x0,
        y0=y0,
        verbose=True,
        callback=[checkpoint_saver],
    )

    plot_convergence(result).figure.savefig("convergence.png")

    print(result)


if __name__ == "__main__":
    main()
