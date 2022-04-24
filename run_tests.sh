#/bin/bash
summit_path=$(python -c "import importlib.util;print(importlib.util.find_spec('summit').submodule_search_locations[0])")
pytest $summit_path $summit_path/../tests --doctest-modules --disable-warnings
