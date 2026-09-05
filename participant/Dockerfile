FROM codercom/code-server:4.133.0
USER root

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates clang \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/local/bin/python
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt
RUN mkdir -p /home/coder/.local/share/code-server/User && printf '{"workbench.colorTheme":"Default Dark Modern","terminal.integrated.cwd":"/home/coder/workshop","remote.autoForwardPorts":false}\n' > /home/coder/.local/share/code-server/User/settings.json \
    && chown -R coder:coder /home/coder/.local
COPY --chown=coder:coder participant/1.py participant/2.py participant/3.py participant/workshop.py participant/INSTRUCTIONS.md /home/coder/workshop/

ENV DEV=CPU
USER coder
WORKDIR /home/coder/workshop

EXPOSE 8080
CMD ["code-server", "--disable-proxy", "--auth", "none", "--bind-addr", "0.0.0.0:8080", "/home/coder/workshop"]
