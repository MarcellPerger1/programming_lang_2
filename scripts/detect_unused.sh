vulture ./parser ./scripts/fuzz.py ./scripts/benchmark.py ./test \
 --ignore-decorators="@_register_autowalk_expr*,@_node_typechecker,@_TypecheckerInitVars.method,@register_corresponding_token"
