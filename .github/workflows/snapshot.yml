name: snapshot

on:
  workflow_dispatch:
    inputs:
      force:
        description: "Guardar sin comprobar la hora ni la sesion (archivo manual, el detector lo ignora)"
        type: boolean
        default: false
  schedule:
    # Lunes a viernes, DESPUES del cierre de EE.UU. (las cotizaciones llevan ~15 min de retraso):
    #   21:50 / 22:10 / 22:25 UTC  =  3:50 / 4:10 / 4:25 pm en El Salvador
    #   invierno: 16:50 / 17:10 / 17:25 ET   |   verano: 17:50 / 18:10 / 18:25 ET
    # El script solo guarda entre las 16:40 y las 18:30 ET; si una ejecucion ya guardo la sesion,
    # las demas no hacen nada. GitHub puede retrasar los horarios: por eso hay tres.
    - cron: "50 21 * * 1-5"
    - cron: "10 22 * * 1-5"
    - cron: "25 22 * * 1-5"

permissions:
  contents: write

concurrency:
  group: snapshot
  cancel-in-progress: false

jobs:
  snap:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: pip install requests pandas

      - name: Descargar y guardar snapshot
        env:
          FORCE: ${{ github.event.inputs.force == 'true' && '1' || '0' }}
        run: python snapshot.py

      - name: Subir al repositorio
        run: |
          git config user.name "snapshot-bot"
          git config user.email "snapshot-bot@users.noreply.github.com"
          git add snapshots
          if git diff --cached --quiet; then
            echo "Nada que guardar en esta ejecucion."
            exit 0
          fi
          git commit -m "snapshot $(date -u +%F_%H%M)"
          git pull --rebase
          git push
