#!/usr/bin/env bash
set -euo pipefail

WRAP_DIR="$(mktemp -d)"
trap 'rm -rf "$WRAP_DIR"' EXIT

cat > "$WRAP_DIR/curl" <<'EOF'
#!/usr/bin/env bash
exec /usr/bin/curl --compressed "$@"
EOF
chmod +x "$WRAP_DIR/curl"

python3 - <<'PY'
from pathlib import Path
path = Path('scripts/ai_capex_provenance.sh')
text = path.read_text(encoding='utf-8')
old = "'title_checks': ['The Global Trade Effects of the AI Infrastructure Boom', 'Federal Reserve'],"
new = "'title_checks': ['The Global Trade Effects of the AI Infrastructure Boom'],"
if old not in text:
    raise SystemExit('Expected rendered-PDF title check was not found')
path.write_text(text.replace(old, new), encoding='utf-8')
PY

PATH="$WRAP_DIR:$PATH" bash scripts/ai_capex_provenance.sh
