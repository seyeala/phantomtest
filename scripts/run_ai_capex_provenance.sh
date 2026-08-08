#!/usr/bin/env bash
set -euo pipefail

WRAP_DIR="$(mktemp -d)"
trap 'rm -rf "$WRAP_DIR"' EXIT

cat > "$WRAP_DIR/curl" <<'EOF'
#!/usr/bin/env bash
exec /usr/bin/curl --compressed "$@"
EOF
chmod +x "$WRAP_DIR/curl"

PATH="$WRAP_DIR:$PATH" bash scripts/ai_capex_provenance.sh
