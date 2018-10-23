for dummy in 1
do
	for model in vector-capsules
	do
		for em_iters in 2 3
		do
			for epochs in 10 15 20
			do
				for batch_size in 64 32 16 8 1
				do
					for weight_decay in 0 0.1 0.25 0.5 0.75 0.9
					do
						for lr in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
						do
                					python3 train.py --model $model --batch-size $batch_size --epochs $epochs --lr $lr --weight-decay $weight_decay --em-iters $em_iters
						done
					done
				done
			done
		done
        done
done

