#!/bin/bash
cd "$(dirname "$0")"
mkdir -p tmp results

# Download missing pseudopotentials
bash download_pp.sh

echo "=== DFT: CeIn3 ==="
/opt/miniforge/envs/qe/bin/pw.x < CeIn3_scf.in > results/CeIn3_scf.out 2>&1
grep "!" results/CeIn3_scf.out && echo "  CONVERGED" || echo "  FAILED"

echo "=== DFT: Sb2Te3 ==="
/opt/miniforge/envs/qe/bin/pw.x < Sb2Te3_scf.in > results/Sb2Te3_scf.out 2>&1
grep "!" results/Sb2Te3_scf.out && echo "  CONVERGED" || echo "  FAILED"

echo "=== DFT: BaBiTe ==="
/opt/miniforge/envs/qe/bin/pw.x < BaBiTe_scf.in > results/BaBiTe_scf.out 2>&1
grep "!" results/BaBiTe_scf.out && echo "  CONVERGED" || echo "  FAILED"

echo "=== DFT: AuPd11 ==="
/opt/miniforge/envs/qe/bin/pw.x < AuPd11_scf.in > results/AuPd11_scf.out 2>&1
grep "!" results/AuPd11_scf.out && echo "  CONVERGED" || echo "  FAILED"

echo "=== DFT: CuIn2K ==="
/opt/miniforge/envs/qe/bin/pw.x < CuIn2K_scf.in > results/CuIn2K_scf.out 2>&1
grep "!" results/CuIn2K_scf.out && echo "  CONVERGED" || echo "  FAILED"

echo "=== ALL DONE ==="
