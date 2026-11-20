#!/bin/sh
# Write the browser-side runtime config from environment variables at container start,
# so one image can point at any backend without a rebuild.
set -eu
escaped=$(printf '%s' "${API_BASE_URL:-}" | sed 's/\\/\\\\/g; s/"/\\"/g')
cat > /usr/share/nginx/html/config.js <<CFG
window.__APP_CONFIG__ = { API_BASE_URL: "${escaped}" };
CFG
echo "runtime config: API_BASE_URL=${API_BASE_URL:-<unset>}"
