#!/bin/bash
cd /usr/share/espresso/pseudo
[ -f "Au.pbe-n-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Au.pbe-n-kjpaw_psl.1.0.0.UPF" -O "Au.pbe-n-kjpaw_psl.1.0.0.UPF" && echo "OK: Au"
[ -f "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Ba.pbe-spn-kjpaw_psl.1.0.0.UPF" -O "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF" && echo "OK: Ba"
[ -f "Bi.pbe-dn-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF" -O "Bi.pbe-dn-kjpaw_psl.1.0.0.UPF" && echo "OK: Bi"
[ -f "Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF" -O "Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF" && echo "OK: Ce"
[ -f "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF" -O "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF" && echo "OK: Cu"
[ -f "In.pbe-dn-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/In.pbe-dn-kjpaw_psl.1.0.0.UPF" -O "In.pbe-dn-kjpaw_psl.1.0.0.UPF" && echo "OK: In"
[ -f "K.pbe-spn-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/K.pbe-spn-kjpaw_psl.1.0.0.UPF" -O "K.pbe-spn-kjpaw_psl.1.0.0.UPF" && echo "OK: K"
[ -f "Pd.pbe-n-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Pd.pbe-n-kjpaw_psl.1.0.0.UPF" -O "Pd.pbe-n-kjpaw_psl.1.0.0.UPF" && echo "OK: Pd"
[ -f "Sb.pbe-n-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Sb.pbe-n-kjpaw_psl.1.0.0.UPF" -O "Sb.pbe-n-kjpaw_psl.1.0.0.UPF" && echo "OK: Sb"
[ -f "Te.pbe-n-kjpaw_psl.1.0.0.UPF" ] || sudo wget -q "https://pseudopotentials.quantum-espresso.org/upf_files/Te.pbe-n-kjpaw_psl.1.0.0.UPF" -O "Te.pbe-n-kjpaw_psl.1.0.0.UPF" && echo "OK: Te"
