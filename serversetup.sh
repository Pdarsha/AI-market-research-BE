#!/bin/bash

set -e

echo "======================================"
echo " Ubuntu Server Setup Started"
echo "======================================"

# Update packages
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install required packages
echo "Installing required packages..."
sudo apt install -y \
    curl \
    git \
    nginx \
    certbot \
    python3-certbot-nginx \
    python3-pip \
    python3-venv \
    build-essential

#############################################
# Install Node.js using NVM
#############################################

install_node() {

    if command -v node >/dev/null 2>&1; then
        echo "Node.js is already installed."
        node -v
        return
    fi

    echo "Installing NVM..."

    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

    export NVM_DIR="$HOME/.nvm"

    if [ -s "$NVM_DIR/nvm.sh" ]; then
        . "$NVM_DIR/nvm.sh"
    fi

    echo
    echo "Choose Node.js installation option"
    echo "1) Install Latest LTS"
    echo "2) Install Specific Version"

    read -p "Enter choice (1 or 2): " choice

    if [ "$choice" = "1" ]; then

        nvm install --lts
        nvm use --lts
        nvm alias default 'lts/*'

    elif [ "$choice" = "2" ]; then

        echo
        echo "Available versions:"
        nvm list-remote

        echo
        read -p "Enter version (Example: 20.19.0): " version

        nvm install "$version"
        nvm use "$version"
        nvm alias default "$version"

    else
        echo "Invalid choice."
        exit 1
    fi

    echo "Installed Node Version:"
    node -v
    npm -v
}

#############################################
# Install PM2
#############################################

install_pm2() {

    if command -v pm2 >/dev/null 2>&1; then
        echo "PM2 already installed."
        pm2 -v
        return
    fi

    echo "Installing PM2..."

    npm install -g pm2

    pm2 -v
}

#############################################
# Configure Nginx
#############################################

configure_nginx() {

    sudo systemctl enable nginx
    sudo systemctl start nginx

    echo "Nginx installed successfully."
}

#############################################
# Configure Certbot
#############################################

configure_certbot() {

    echo "Certbot installed successfully."
}

#############################################
# Main
#############################################

install_node

install_pm2

configure_nginx

configure_certbot

echo
echo "======================================"
echo " Server setup completed successfully!"
echo "======================================"

echo
echo "Installed Versions:"
echo "-----------------------------"

node -v
npm -v
pm2 -v
python3 --version

echo
echo "Next Steps:"
echo "1. Clone your project"
echo "2. Create Python virtual environment"
echo "3. Install requirements"
echo "4. Configure .env"
echo "5. Start using:"
echo
echo "   pm2 start app.yaml"
echo
echo "6. Save PM2:"
echo
echo "   pm2 save"
echo "   pm2 startup"