# A programming language I'm making
### Aim(s)
- This is mainly so that I learn how to make a programming language (esp. how to make the parser as that seems like the trickiest to do without using a parser generator)
- This might evetually be used for my other project, [`pymeowlib`](https://github.com/MarcellPerger1/pymeowlib) (compiling python or some othher text-based language to scratch)
  so most of the keywords and similar have a 1-to-1 scratch equivalent.
  This language would eventually be used:
  - either as an internal representaion (e.g. a function for each of the opcodes of python) so I don't have to implement everythin using drag-and-drop
  - or as the language that can be compiled to scratch

### Code style/guidelines
- Rule 0: **`readability counts`**.
- I generally follow [PEP 8](https://peps.python.org/pep-0008/) except:
- Line length:
  - Lines longer than 93 characters should be wrapped
  - When wrapping lines, each component should be at most 80 characters.
  - i.e. If a statement is over multiple lines, max length is 80 If it's a single line, max length is 93.
- Quotes:
  - Generally use double quotes (`"`), especially for error message and similar (makes it easier to add `'` if the error message is tweaked).
  - Single quotes (`'`) may be used for internal IDs and strings not directly displayed to the used (e.g. `'<='`, `'call_args'`)
- Private names (`__private`):
  - Don't use them. They cause problems with getattr and similar annoyances.
- Indentation in long wrapped conditions (`if`/`while`):
  - Add an extra indent (4 spaces) to distinguish the 2nd/3rd line of the condition from the body of the block.
- Type annotations:
  - Always use them for parameter types, even in internal/private functions.
  - Use them for return types if the IDE (Pycharm for me) can't figure it out, or if it's unclear, or you just feel like it.
  - Even use them for local varaibles (if the IDE can't infer the types - e.g. the typechecker doesn't know what type goes in the list if it's `ls = []`)
  - Use the most accurate/precise types possible (e.g. `dict[str | tuple[str, int], type[NamedNodeCls]` instead of `dict` (which gives almost no information))
  - If type are getting too long/repetitive/recursive, use type variables.
  - Try to make typing work without resorting to `Any` (e.g. using `cast`).
  - `from __future__ import annotations` + `if TYPE_CHECKING` is good if you expereince circular import issues due to types.
- `assert`s:
  - Highly encouraged to describe assumptions (e.g. about a function's inputs) - as long as it isn't way too many of them.
  - Don't `assert` the type of arguments - types should be specfied in the annotations (these can be considered assertions as IDEs/type checkers should catch violations of these).
  - Don't `assert` if it massively deteriorates performance (e.g. don't check every element in the list if you aren't already iterating over it) - put a comment instead if the assumption is non-obvious.
  - Only use assert for internal assumptions. Don't use it as a shortcut to raising errors (e.g. don't use it to raise syntax errors in the source code).
- Keyword arguments:
  - Pass an argument as keyword arg (e.g. `foo(bar=True)`) if it would otherwise be unclear what it is doing.
  - (e.g. `tformat(obj, 1, True, False)` is very unclear/not very readable. `tformat(obj, indent=1, verbose=True, append_lf=False)` is much better, makes sense even to people who haven't seen this codebase before).
- Wildcard imports
  - Importing from outside this project: no (we don't know what extra stuff other projects put in their module namespace that might break our code).
  - Importing internally within the project:
    - Useful when you need *everything* from a module that has many *similar* classes
    - (e.g. importing all the nodes (`parser/cst/nodes.py`) in the CST generation code (`parser/cst/treegen.py`) is good).
    - The threshold for 'many' here is around 10-15 classes.
    - Otherwise (if the classes/functions are unrelated (e.g. in `utils.py`), or you don't need all/most of them, or the module has few classes), just list them out.
    - You must not `import *` from a module without a `__all__` (to avoid polluting your globals with their internal stuff, like the stuff they import from elsewhere). The one exception to this is re-exporting stuff (e.g. in `__init__.py`).
- In general, be sensible, use your judgement as to what makes it more readable. If in doubt, just choose one, don't waste time on formtting that is easy to change later.
- Good code with bad formatting is much better than bad code with good formatting (although good code with good formatting is preferable to both).
