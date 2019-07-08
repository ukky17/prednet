for i in Ahat0 Ahat1 Ahat2 Ahat3 A0 A1 A2 A3 E0 E1 E2 E3 R0 R1 R2 R3
do
  for j in MAE_P_deg0 MAE_P_deg180
  do
    python predict.py --nt 50 --target $i --stim $j
  done
done
