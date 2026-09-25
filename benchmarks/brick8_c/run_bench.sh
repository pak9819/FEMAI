#!/bin/zsh
# Fairer Laufzeitvergleich brick8 analytisch vs. KI in kompiliertem C.
#   ./run_bench.sh brick8_nl_W_network_h32d3.mat [weitere Netze ...]
# Fuer jedes Netz: Gewichte exportieren, MATLAB-Referenz (Ke/Finte beider
# Elemente) erzeugen, C-Ergebnis dagegen pruefen und Zeiten messen, einmal
# mit einfachen Schleifen und einmal mit Accelerate-BLAS im MLP.
# Die CPU sollte frei sein (kein Training parallel).
set -e
HERE=${0:A:h}
ROOT=${HERE:h:h}
MATLAB=/Applications/MATLAB_R2026a.app/bin/matlab
OUT=$HERE/out
mkdir -p $OUT
clang -O3 -mcpu=native -ffp-contract=fast -o $OUT/brick8_bench $HERE/brick8_bench.c -lm
clang -O3 -mcpu=native -ffp-contract=fast -DUSE_BLAS -o $OUT/brick8_bench_blas \
      $HERE/brick8_bench.c -framework Accelerate -lm
for NET in "$@"; do
  TAG=${NET:r}
  $ROOT/.venv/bin/python $HERE/export_c_weights.py $ROOT/sourcecode/elements/brick8/$NET $OUT/$TAG.bin
  $MATLAB -batch "cd('$ROOT'); startup; addpath('$HERE'); warning('off','element_brick8_nl_ai:OutOfHull'); export_c_reference('$NET','$OUT/${TAG}_ref.bin'); matlab_element_timing('$NET');" 2>&1 | grep -E "geschrieben|MATLAB "
  $OUT/brick8_bench      $OUT/$TAG.bin $OUT/${TAG}_ref.bin 200
  $OUT/brick8_bench_blas $OUT/$TAG.bin $OUT/${TAG}_ref.bin 200
done
