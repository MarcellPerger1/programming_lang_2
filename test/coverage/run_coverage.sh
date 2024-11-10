coverage run -m unittest discover -s ./test/ -t ./ "$@" &&
 coverage html &&
 python ./test/coverage/inject_colors.py
