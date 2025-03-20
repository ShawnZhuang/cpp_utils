mkdir -p ~/.pip/
touch  ~/.pip/pip.conf

cat <<EOL > ~/.pip/pip.conf
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
EOL
