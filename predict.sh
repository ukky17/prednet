for i in E0 E1 E2 E3 R0 R1 R2 R3
do
    for j in MAE_P_deg0 MAE_P_deg180
    do
        python predict.py --nt 50 --model_type Lall \
                          --target $i --stim $j
    done
done

for i in E0 E1 E2 E3 R0 R1 R2 R3
do
    for j in MAE_P_deg0 MAE_P_deg180
    do
      python predict.py --nt 10 --model_type Lall \
                        --target $i --stim $j
    done
done
