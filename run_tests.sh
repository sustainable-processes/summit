#/bin/bash
# package_path=$(python -c "import sys;print(sys.prefix)")
# summit_path="$package_path/site-packages/summit"
summit_path=$(python -c "import importlib.util;print(importlib.util.find_spec('summit').submodule_search_locations[0])")
pytest $summit_path $summit_path/../tests --doctest-modules --disable-warnings
