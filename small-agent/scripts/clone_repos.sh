#!/usr/bin/env bash
# Clone small open-source repositories for SFT data generation
# Make executable with: chmod +x clone_repos.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOS_DIR="${SCRIPT_DIR}/../repos"

mkdir -p "${REPOS_DIR}"
cd "${REPOS_DIR}"

echo "Cloning repositories to ${REPOS_DIR}..."

# Python repos (8)
[ -d "httpie" ] || git clone --depth 1 https://github.com/httpie/httpie.git
[ -d "click" ] || git clone --depth 1 https://github.com/pallets/click.git
[ -d "requests" ] || git clone --depth 1 https://github.com/psf/requests.git
[ -d "httpx" ] || git clone --depth 1 https://github.com/encode/httpx.git
[ -d "fastapi" ] || git clone --depth 1 https://github.com/tiangolo/fastapi.git
[ -d "starlette" ] || git clone --depth 1 https://github.com/encode/starlette.git
[ -d "flask" ] || git clone --depth 1 https://github.com/pallets/flask.git
[ -d "pip" ] || git clone --depth 1 https://github.com/pypa/pip.git

# Rust repos (5)
[ -d "ripgrep" ] || git clone --depth 1 https://github.com/BurntSushi/ripgrep.git
[ -d "fd" ] || git clone --depth 1 https://github.com/sharkdp/fd.git
[ -d "bat" ] || git clone --depth 1 https://github.com/sharkdp/bat.git
[ -d "hyperfine" ] || git clone --depth 1 https://github.com/sharkdp/hyperfine.git
[ -d "just" ] || git clone --depth 1 https://github.com/casey/just.git

# JS/TS repos (6)
[ -d "express" ] || git clone --depth 1 https://github.com/expressjs/express.git
[ -d "koa" ] || git clone --depth 1 https://github.com/koajs/koa.git
[ -d "got" ] || git clone --depth 1 https://github.com/sindresorhus/got.git
[ -d "chalk" ] || git clone --depth 1 https://github.com/chalk/chalk.git
[ -d "commander.js" ] || git clone --depth 1 https://github.com/tj/commander.js.git
[ -d "yargs" ] || git clone --depth 1 https://github.com/yargs/yargs.git

# Go repos (4)
[ -d "fzf" ] || git clone --depth 1 https://github.com/junegunn/fzf.git
[ -d "lazygit" ] || git clone --depth 1 https://github.com/jesseduffield/lazygit.git
[ -d "bubbletea" ] || git clone --depth 1 https://github.com/charmbracelet/bubbletea.git
[ -d "lipgloss" ] || git clone --depth 1 https://github.com/charmbracelet/lipgloss.git

echo "Done! Cloned 23 repositories."
