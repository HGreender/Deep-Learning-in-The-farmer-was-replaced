from Movements import move_to_start
from NeuroNet import *
from Dataset import h_w_f_g


def main():
	X = []
	y = []
	for row in h_w_f_g:
		height_cm = row[0]
		weight_kg = row[1]
		age = row[2]
		gender = row[3]
		
		height_m = height_cm / 100.0

		if height_m > 0:
			bmi = weight_kg / (height_m * height_m)
		else:
			bmi = 0.0
		
		X.append([bmi, age])
	
		if gender == 'M':
			label = 1
		else:
			label = 0
		y.append(label)
	
	test_size = 0.3
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)

	mean, std = compute_mean_std(X_train)
	X_train_norm = normalize(X_train, mean, std)
	X_test_norm = normalize(X_test, mean, std)
	
	hidden_layers_count = 2
	hidden_size = 8
	output_size = 2
	epochs = 300
	lr = 0.15
	batch_size = 8
	activation_function = 'relu'

	model = train_best_loss_network(
		X_train_norm, y_train,
		hidden_layers_count, hidden_size,
		output_size,
		epochs, lr,
		batch_size,
		activation_function
	)

	if model != None:
		probs = predict(X_test_norm, model, activation_function)
		y_pred = []
		for i in range(len(probs)):
			if probs[i][0] > probs[i][1]:
				y_pred.append(0)
			else:
				y_pred.append(1)

		correct = 0
		total = len(y_test)
		for i in range(total):
			if y_test[i] == y_pred[i]:
				correct += 1
		accuracy = correct / total

		print("Результаты:")
		sleep(1)
		print("Точность: ", accuracy * 100, "%")
		quick_print(model)


if __name__ == "__main__":
	clear()
	move_to_start()
	main()