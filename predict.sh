for i in Ahat0 Ahat1 Ahat2 A0 A1 A2 E0 E1 E2 R0 R1 R2
do
  for j in MAE_P_deg0 MAE_P_deg180
  do
    python predict.py --target $i --stim $j --SAVE_DIR './response/190826_1/' --WEIGHTS_DIR './model/190826_1/' --nt 230
  done
done
