========== START GRADING
----- START 2a-0-basic:  Word2vec sanity check 1
==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====
Gradient check passed!

			Skip-Gram with naive_softmax_loss_and_gradient			

Your Result:
Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]

Expected Result: Value should approximate these:
Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]

----- END 2a-0-basic [took 0:00:00.166731 (max allowed 5 seconds), 1/1 points]

----- START 2a-1-basic:  Word2vec sanity check 2
			naive_softmax_loss_and_gradient			

Your Result:
Loss: 2.217424879078895
Gradient wrt Center Vector (dJ/dV):
 [-0.17249875  0.64873661  0.67821423]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.11394934  0.05228819  0.39774391]
 [-0.02740743  0.01257651  0.09566654]
 [-0.03385715  0.01553611  0.11817949]
 [ 0.24348396 -0.11172803 -0.84988879]
 [-0.06827005  0.03132723  0.23829885]]

Expected Result: Value should approximate these:
Loss: 2.217424877675181
Gradient wrt Center Vectors(dJ/dV):
 [-0.17249875  0.64873661  0.67821423]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.11394933  0.05228819  0.39774391]
 [-0.02740743  0.01257651  0.09566654]
 [-0.03385715  0.01553611  0.11817949]
 [ 0.24348396 -0.11172803 -0.84988879]
 [-0.06827005  0.03132723  0.23829885]]

----- END 2a-1-basic [took 0:00:00.000302 (max allowed 5 seconds), 1/1 points]

----- START 2a-2-basic:  Word2vec sanity check 3
			test sigmoid			

Your Result:
[0.38553435 0.29385824 0.6337228  0.40988622 0.29385824 0.58343371
 0.29385824 0.58343371 0.40988622 0.40988622 0.6337228 ]
Expected Result: Value should approximate these:
[0.38553435 0.29385824 0.63372281 0.40988622 0.29385824 0.5834337
 0.29385824 0.5834337  0.40988622 0.40988622 0.63372281]
----- END 2a-2-basic [took 0:00:00.000122 (max allowed 5 seconds), 1/1 points]

----- START 2a-3-hidden:  Sigmoid with 1D inputs
----- END 2a-3-hidden [took 0:00:00.000045 (max allowed 15 seconds), ???/1 points] (hidden test ungraded)

----- START 2a-4-hidden: Sigmoid with 2D inputs
----- END 2a-4-hidden [took 0:00:00.000036 (max allowed 15 seconds), ???/1 points] (hidden test ungraded)

----- START 2a-5-hidden:  Sigmoid with large 2D inputs
----- END 2a-5-hidden [took 0:00:00.021164 (max allowed 15 seconds), ???/2 points] (hidden test ungraded)

----- START 2a-6-hidden:  test softmax
----- END 2a-6-hidden [took 0:00:00.000225 (max allowed 10 seconds), ???/2 points] (hidden test ungraded)

----- START 2a-7-hidden:  test skipgram
----- END 2a-7-hidden [took 0:00:00.000213 (max allowed 10 seconds), ???/3 points] (hidden test ungraded)

----- START 2b-0-basic:  SGD sanity check 1
iter 100: 0.004578
iter 200: 0.004353
iter 300: 0.004136
iter 400: 0.003929
iter 500: 0.003733
iter 600: 0.003546
iter 700: 0.003369
iter 800: 0.003200
iter 900: 0.003040
iter 1000: 0.002888
test 1 result: 8.414836786079764e-10
----- END 2b-0-basic [took 0:00:00.001798 (max allowed 5 seconds), 0.5/0.5 points]

----- START 2b-1-basic:  SGD sanity check 2
iter 100: 0.000000
iter 200: 0.000000
iter 300: 0.000000
iter 400: 0.000000
iter 500: 0.000000
iter 600: 0.000000
iter 700: 0.000000
iter 800: 0.000000
iter 900: 0.000000
iter 1000: 0.000000
test 2 result: 0.0
----- END 2b-1-basic [took 0:00:00.001774 (max allowed 5 seconds), 0.5/0.5 points]

----- START 2b-2-basic:  SGD sanity check 3
iter 100: 0.041205
iter 200: 0.039181
iter 300: 0.037222
iter 400: 0.035361
iter 500: 0.033593
iter 600: 0.031913
iter 700: 0.030318
iter 800: 0.028802
iter 900: 0.027362
iter 1000: 0.025994
test 3 result: -2.524451035823933e-09
----- END 2b-2-basic [took 0:00:00.001769 (max allowed 5 seconds), 1/1 points]

----- START 2b-3-hidden:  sgd quad scalar
----- END 2b-3-hidden [took 0:00:00.001774 (max allowed 5 seconds), ???/1 points] (hidden test ungraded)

----- START 2b-4-hidden:  sgd quad matrix
----- END 2b-4-hidden [took 0:00:00.009872 (max allowed 5 seconds), ???/1 points] (hidden test ungraded)

----- START 2c-0-basic:  Sanity check for word2vec implementation
<class 'Exception'>
Excecute run.py to generate sample_vectors_(soln).json
----- END 2c-0-basic [took 0:00:00.000347 (max allowed 5 seconds), 0/1 points]

----- START 2c-1-hidden:  Compare word vector outputs (sample_vectors_soln.json) with solution.
<class 'AssertionError'>
False is not true : Cannot run unit test because word vector file is not present.  It must be uploaded with your submission.py file.  Execute src/run.py to create the word vector file (sample_vectors_soln.json).
----- END 2c-1-hidden [took 0:00:00.000078 (max allowed 300 seconds), ???/3 points] (hidden test ungraded)

Note that the hidden test cases do not check for correctness.
They are provided for you to verify that the functions do not crash and run within the time limit.
Points for these parts not assigned by the grader unless the solution is present (indicated by "???").
========== END GRADING [5.0/6.0 points + 0/0 extra credit]
