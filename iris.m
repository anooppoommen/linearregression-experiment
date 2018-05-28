clear % erase all variables
load './dataset/iris.txt' 

%feature matrix X
X = iris(:,1:4);
%label matrix
Y = iris(:,5);

%size of training examples
m = size(X)(1);

%parametre matrix
theta = zeros(5,1);

iteration    = 100;
learningRate = 0.01;

jHistory     = zeros(1,iteration);

for i = [1:iteration]
	H = [ones(m,1),X] * theta;
	J =  (1/(2*m) ) * sum((H-Y).^2);
	jHistory(i) = J;
	printf("Current cost is %f\n",J);
	%appliying gradient descent

	update = theta - learningRate * (1/m) *([ones(m,1),X]' * (H-Y)) 
	theta = update
endfor

%test prediction
predicted1 = sum([1 5.7 4.4 1.5 0.4]*theta) %expects val close to 1
predicted2 = sum([1 5.8 2.7 4.1 1.0]*theta) %expects val close to 2
predicted2 = sum([1 5.8 2.7 5.1 1.9]*theta) %expects val close to 3