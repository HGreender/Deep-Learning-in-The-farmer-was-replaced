from Movements import move_to_start
from NeuroNet import *
from Utils import sleep
from weghts import saved_weght_acc_00
from Dataset import h_w_f_g


def main():
	X = []
	y = []
	for row in h_w_f_g:
		height_cm = row[0]
		weight_kg = row[1]
		shoe_size = row[2]
		gender = row[3]
		
		height_m = height_cm / 100.0

		if height_m > 0:
			bmi = weight_kg / (height_m * height_m)
		else:
			bmi = 0.0
		
		X.append([bmi, shoe_size])
	
		if gender == 'M':
			label = 1
		else:
			label = 0
		y.append(label)
	
	test_size = 0.3
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)

	mean, std = compute_mean_std(X_train)
	data = [[181, 55, 43]]
	height_cm = data[0][0]
	weight_kg = data[0][1]
	shoe_size = data[0][2]
	height_m = height_cm / 100.0
	if height_m > 0:
		bmi = weight_kg / (height_m * height_m)
	else:
		bmi = 0.0
	normalized_data = [[bmi, shoe_size]]
	X_test_norm = normalize(normalized_data, mean, std)

	activation_function = 'relu'

	model = saved_weght_acc_00
	
	probs = predict(X_test_norm, model, activation_function)
	y_pred = []
	for i in range(len(probs)):
		if probs[i][0] > probs[i][1]:
			y_pred.append("Девушка")
		else:
			y_pred.append("Мужчина")
	print(y_pred)


if __name__ == "__main__":
	clear()
	move_to_start()
	main()