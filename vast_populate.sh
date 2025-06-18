# --------- paste everything between the lines ----------

# stop on errors, undefined vars, or pipe failures
set -euo pipefail

# ⬇️ 1) ensure your personal access token is set
: "${GITHUB_TOKEN:?Please export GITHUB_TOKEN first, e.g.  export GITHUB_TOKEN=ghp_xxx}"

# ⬇️ 2) clone the repo quietly (shallow clone for speed)
git clone --quiet --depth 1 "https://${GITHUB_TOKEN}@github.com/Dene33/amazon-hidden-site-hunter.git"

# ⬇️ 3) install uv without prompts
curl -LsSf https://astral.sh/uv/install.sh | CI=1 sh   # CI=1 suppresses any confirmation

# ⬇️ 4) load the environment file the installer just wrote
# shellcheck disable=SC1090
source "$HOME/.local/bin/env"

# ⬇️ 5) move into the project directory
cd amazon-hidden-site-hunter

# ⬇️ 6) sync project dependencies (non-interactive)
"$HOME/.local/bin/uv" sync

# --------- end of block ----------
