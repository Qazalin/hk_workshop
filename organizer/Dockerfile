FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends clang git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd --home-dir /tmp --uid 10001 workshop
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /opt/workshop
COPY organizer/server.py ./

ENV DEV=CPU
USER workshop
ENTRYPOINT ["python", "/opt/workshop/server.py", "eval", "/submission/solution.py"]
