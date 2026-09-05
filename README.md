# tinygrad CPU workshop

`participant/` is the handout and browser editor environment. `organizer/` is the evaluator and scoreboard.
Run these commands from this folder:

```sh
docker build --platform linux/amd64 -f participant/Dockerfile -t hk-workshop .
docker build --platform linux/amd64 -f organizer/Dockerfile -t hk-workshop-eval .
export WORKSHOP_ADMIN_TOKEN=organizer-token
python organizer/server.py serve --host 0.0.0.0 --cpuset 0
```

Run a participant environment:

```sh
docker run --rm -p 8080:8080 --add-host=host.docker.internal:host-gateway \
  -e WORKSHOP_SERVER=http://host.docker.internal:3000 hk-workshop
```

Note: change WORKSHOP_SERVER to actual domain assigned to the organizer server on workshop day.

Open http://localhost:8080 for the editor and http://localhost:3000 for the scoreboard.

Organizer starts challenges one at a time:

```sh
python organizer/server.py start 1
```

Participants submit using the instructions provided in `participant/README.md`

The evaluator on organizer server processes submissions serially in a temp container.
