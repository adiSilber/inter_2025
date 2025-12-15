umask 077
mkdir -p ~/.secrets.d && chmod 700 ~/.secrets.d

cat > ~/.secrets.d/env.sh <<'EOF'
export HF_TOKEN=""
EOF
chmod 600 ~/.secrets.d/env.sh

grep -qxF '[ -f ~/.secrets.d/env.sh ] && . ~/.secrets.d/env.sh' ~/.bashrc \
  || echo '[ -f ~/.secrets.d/env.sh ] && . ~/.secrets.d/env.sh' >> ~/.bashrc
