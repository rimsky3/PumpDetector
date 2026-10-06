#!/bin/bash
# Vultr Startup Script (type: Boot), runs once as root on first boot.
# ASCII only: Vultr accepts Latin-1 characters only.
# 1) SSH also on port 2222 - the owner's ISP blocks outbound port 22;
# 2) user "ubuntu" with the same key as on Lightsail (D:\keys\pd.pem), so deploy.sh / pull_db.sh work unchanged.
# Only the PUBLIC key is here (safe to show); the private key stays on the laptop.
set -e
PUBKEY='ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC2f7cNa5K76S5GElRCCKz4GtxqkzSPSZCJGM8WHIiUPYGDLR524lwqe3z+pxUDUX9DS+Qft0PpE6ZsmSxl1hkfxHsoX5L0Tu42kwJbkKXpdvA/ddxQJ7WEW4jp4qfuEhUTkvVu+7D40JFdJN4lqJt+o1Zt0xanCNaVAVWB3ecVL+zK5IZFjLF7yjotWKSRgyvz7vPQl4y/9B6Dsi04kUz+T5q0MIfftrc7pgcGnRmE0RQjrH2BgoEal6axwO6jPMAE2NUj63Ckze13j/0VVczby0AyordArTMnkVY2DWSq3J9RxlbBeRrd8V+X8a6iC+w31ouZ5wsD/teEwvpV6eMZ pump-detector'

# user ubuntu: passwordless sudo (login by key only)
id ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash ubuntu
usermod -aG sudo ubuntu
echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-ubuntu
chmod 440 /etc/sudoers.d/90-ubuntu
install -d -m 700 -o ubuntu -g ubuntu /home/ubuntu/.ssh
echo "$PUBKEY" > /home/ubuntu/.ssh/authorized_keys
chmod 600 /home/ubuntu/.ssh/authorized_keys
chown ubuntu:ubuntu /home/ubuntu/.ssh/authorized_keys

# SSH on 22 and 2222. Ubuntu 24.04 runs SSH either via a systemd socket or as a service - configure both
mkdir -p /etc/ssh/sshd_config.d
printf 'Port 22\nPort 2222\n' > /etc/ssh/sshd_config.d/10-ports.conf
if systemctl is-enabled ssh.socket >/dev/null 2>&1; then
  mkdir -p /etc/systemd/system/ssh.socket.d
  printf '[Socket]\nListenStream=\nListenStream=0.0.0.0:22\nListenStream=[::]:22\nListenStream=0.0.0.0:2222\nListenStream=[::]:2222\n' \
    > /etc/systemd/system/ssh.socket.d/ports.conf
  systemctl daemon-reload
  systemctl restart ssh.socket
else
  systemctl restart ssh
fi

# Ubuntu firewall (Vultr enables ufw with only port 22 allowed)
if command -v ufw >/dev/null 2>&1; then
  ufw allow 2222/tcp || true
fi
