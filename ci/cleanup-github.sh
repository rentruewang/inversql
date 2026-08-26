# Copyright (c) InverSQL Authors - All Rights Reserved

echo "Removing files we did not ask for..."

echo "Pruning android..."
sudo rm -rf /usr/local/lib/android

echo "Pruning dotnet..."
sudo rm -rf /usr/share/dotnet

echo "Pruning ghcup..."
sudo rm -rf /usr/local/.ghcup

echo "Pruning docker..."
docker system prune -af --volumes

echo "Investigating how much storage is used in GitHub Actions..."
df -h
