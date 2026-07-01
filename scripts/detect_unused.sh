vulture ./parser ./scripts/fuzz.py ./scripts/fuzz.py ./test --ignore-decorators="@_register_autowalk_expr*" --exclude="parser/astgen/ast_print.py"
