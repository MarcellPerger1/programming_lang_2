coverage run --concurrency=multiprocessing,thread -p -m unittest discover -s ./test/ -t ./ "$@" &&
 coverage combine &&
 coverage html &&
 python ./test/coverage/inject_colors.py
