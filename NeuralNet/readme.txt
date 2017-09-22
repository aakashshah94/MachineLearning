Language used is Python:

1> Pre-processing
(preprocessing.py)
How To run:
=>On the command prompt type the path of the interpreter, path of the pyhton file, complete path of input dataset, complete output path of the pre-processed data set  

for example:
"C:\Users\Aakash shah\AppData\Local\Programs\Python\Python36-32\python.exe" "C:\Users\Aakash shah\Documents\aakash study\graduate\sem-2\ml mine\Assignment3\part II\preprocessing.py" "C:\Users\Aakash shah\Downloads\iris.data" "C:\Users\Aakash shah\Documents\aakash study\graduate\sem-2\ml mine\Assignment3\part II\iris_processed.data"	

2>Trainig Neuralnet:
(neuralNet.py)

How To run:
=>On the command prompt type the path of the interpreter
path of the pyhton file
complete path of the post-processed input dataset
training percent 
error tolerance 
number of hidden layers 
number of neurons in each hidden layer  

for example:
"C:\Users\Aakash shah\AppData\Local\Programs\Python\Python36-32\python.exe" "C:\Users\Aakash shah\Documents\aakash study\graduate\sem-2\ml mine\Assignment3\part II\neuralNet.py" "C:\Users\Aakash shah\Documents\aakash study\graduate\sem-2\ml mine\Assignment3\part II\housing_processed.data" 80 0.01 4 2 4 3 2




	
Warning: As cross validation is implemented it takes more time.
         espcially adult dataset and housing data set requires more time.
		 Use 80 20 split. Split of 90 10 will require more time as there will be 10 folds instead of 5.
         
        MAXIMUM NUMBERS OF HIDDEN LAYERS ALLOWED:5		 
        AS CROSS VALIDATION IS USED ADULT AND HOUSING DATASET WILL TAKE 10-15 MINS TO RUN.