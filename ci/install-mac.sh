# Copyright (c) InverSQL Authors - All Rights Reserved

command -v brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

command -v pipx || brew install pipx -y

command -v pdm || pipx install pdm
