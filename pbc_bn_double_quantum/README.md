This folder is based on the folder more_layer_double_quantum
with batch normalization layers
under PBC
1. python genData_qt.py N: data generation
2. python train_qt_dqnn.py  (parameters)
 for example, python train_qt_dqnn.py    1000 50 0.9 0 10 10
3. test_qt_dqnn.py or test_loop_over_epoch_num.py to compute test loss
4. plot loss during training using plt_qt_log.py
5. plt_one_N.py or plt_one_epoch_one_N.py to plot test error  vs linear model error
6. stats_qt_trainData.py to get stats of test data
7. large_lattice_test_qt_dqnn.py to test on larger lattice
8. all_N_all_layer_large_lattice_test_qt_dqnn.py: run test on larger lattice in 1 run
9. fit_stats.py to fit data for larger size
9. plt_coefs.py, to get visualization of layers