{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    uv
    git
  ];

  # Without LD_LIBRARY_PATH, `import tensorflow` fails on NixOS.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
    with pkgs;
    [
      stdenv.cc.cc.lib
      zlib
    ]
  );

  shellHook = ''
    export PYTHONPATH=$PWD:$PYTHONPATH

    # Detect stale/broken venvs
    if [ -d .venv ] && ! .venv/bin/python --version >/dev/null 2>&1; then
      echo ">> Stale .venv detected (broken interpreter) — recreating"
      rm -rf .venv
    fi

    if [ ! -d .venv ]; then
      echo ">> Creating .venv (Python 3.11)"
      uv venv .venv --python ${pkgs.python311}/bin/python3.11
    fi
    source .venv/bin/activate

    if [ ! -f .venv/.synced ] || [ requirements.txt -nt .venv/.synced ]; then
      echo ">> Syncing dependencies from requirements.txt"
      uv pip install -r requirements.txt
      touch .venv/.synced
    fi

    echo ""
    echo "pprl-attack-poc dev shell ready."
    echo "  python : $(python --version)"
    echo "  venv   : $VIRTUAL_ENV"
    echo ""
  '';
}
