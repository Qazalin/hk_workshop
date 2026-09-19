#!/bin/sh
set -eu

mkdir -p /tmp/nginx/client_body /tmp/nginx/proxy /tmp/nginx/fastcgi /tmp/nginx/uwsgi /tmp/nginx/scgi
code-server --disable-proxy --auth none --bind-addr 127.0.0.1:8081 /home/coder/workshop &
exec nginx -c /etc/nginx/nginx.conf -g "daemon off;"
