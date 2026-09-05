FROM codercom/code-server:4.133.0
USER root

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates clang \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/local/bin/python
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt
RUN mkdir -p /home/coder/.local/share/code-server/User && printf '{"workbench.colorTheme":"Default Dark Modern","terminal.integrated.cwd":"/home/coder/workshop","remote.autoForwardPorts":false}\n' > /home/coder/.local/share/code-server/User/settings.json \
    && chown -R coder:coder /home/coder/.local
COPY --chown=coder:coder 1.py 2.py 3.py workshop.py README.md /home/coder/workshop/

ENV DEV=CPU WORKSHOP_SERVER=https://hk-workshop.vercel.app
USER coder
WORKDIR /home/coder/workshop

EXPOSE 8080
CMD ["code-server", "--disable-proxy", "--auth", "none", "--bind-addr", "0.0.0.0:8080", "/home/coder/workshop"]
