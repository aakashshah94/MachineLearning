Language used is Python:

How To run:
1>If the Python Interpreter path is set:
  Then just type in the path and filename in the commandprompt and also pass path of training data,validation data,testing data and pruning factor.

for example
"C:\Users\Aakash shah\Documents\aakash study\graduate\sem-2\ml mine\Assignment 2\Part ll\Dec_tree.py" "C:\Users\Aakash shah\Downloads\data_sets1\training_set.csv" "C:\Users\Aakash shah\Downloads\data_sets1\validation_set.csv" "C:\Users\Aakash shah\Downloads\data_sets1\test_set.csv" 0.02

2>If the Python Interpreter path is not set:
   You will have to pass the path of interpreter before passing the arguments mentioned above. to run the program in this way you will have to change the program slightly.In the main funtion where arguments are passed , just increase the argument number just like following.
 Line: 288
    training = args[0]                    training = args[1]
    validation = args[1]       =>         validation = args[2] 
    test = args[2]                        test = args[3]
    factor = float(args[3])               factor = float(args[4])

	
	If you want to try to get better accuracy for the same pruning factor just change the accuracy threshold as follows.
line:21	accThresh = 0   increase the number to increse accuracy

Warning: Higher numbers may require more running time and processing power.
         Do not use pruning factor greater than 0.5(higher factors may take too long to run).           