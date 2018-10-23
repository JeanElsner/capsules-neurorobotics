for epochs in 10 15 20
do
	for lr in 0.005
	do
		for update_interval in 100 250 500
		do
			for lr_decay in 0.7 0.5 
			do
				for decay_memory in 3 5 10
				do
					for time in 5 10 15
					do
						for n_hidden in 50 100 150 200 250 300 350 400 450 500 600 700 800 900 1000
						do
                					python3 snn_backprop.py --test --n_hidden $n_hidden --time $time --update_interval $update_interval --lr_decay $lr_decay --epochs $epochs --decay_memory $decay_memory --lr $lr
						done
					done
				done
			done
		done
        done
done

