from math import *


def val_shape(array):
	if len(array) == 0:
		return
	const_row_len = len(array[0])
	for row in array:
		if len(row) != const_row_len:
			print('ShapeError: rows len must be equal')
			return


def shape(array):
	if len(array) == 0:
		return 0, 0
	val_shape(array)
	rows_count = len(array)
	cols_count = len(array[0])
	return rows_count, cols_count


def init_network(input_size, hidden_layers_count, hidden_size, output_size):
	layers = []
	current_size = input_size

	for _ in range(hidden_layers_count):
		W = create_random_array(hidden_size, current_size)
		b = []
		for _ in range(hidden_size):
			b.append(0.0)
		layers.append({'W': W, 'b': b})
		current_size = hidden_size

	W_out = create_random_array(output_size, current_size)
	b_out = []
	for _ in range(output_size):
		b_out.append(0.0)
	output_layer = {'W': W_out, 'b': b_out}

	return {
		'layers': layers,
		'output_layer': output_layer
	}


# ========================
# Прямой проход
# ========================

def forward_pass(X, network, activation_function):
	current_input = X
	layers = network['layers']
	output_layer = network['output_layer']

	activations = [X]
	z_values = []

	for layer in layers:
		W = layer['W']
		b = layer['b']
		W_T = transpose(W)
		Z = matmul(current_input, W_T)
		if Z == None:
			print("Ошибка в matmul (скрытый слой)")
			return None, None
		Z = add_bias_matrix(Z, b)
		if activation_function == 'sigmoid':
			A = sigmoid_matrix(Z)
		elif activation_function == 'relu':
			A = relu_matrix(Z)
		else:
			A = relu_matrix(Z)
		z_values.append(Z)
		activations.append(A)
		current_input = A

	W_out = output_layer['W']
	b_out = output_layer['b']
	W_out_T = transpose(W_out)
	Z_out = matmul(current_input, W_out_T)
	if Z_out == None:
		print("Ошибка в matmul (выходной слой)")
		return None, None
	Z_out = add_bias_matrix(Z_out, b_out)
	probs = softmax_batch(Z_out)

	context = {
		'network': network,
		'activations': activations,
		'z_values': z_values,
		'z_output': Z_out,
		'activation_function': activation_function
	}

	return probs, context


# ========================
# Подготовка данных
# ========================

def to_one_hot(y, num_classes):
	one_hot = []
	for i in range(len(y)):
		row = []
		for c in range(num_classes):
			if c == y[i]:
				row.append(1.0)
			else:
				row.append(0.0)
		one_hot.append(row)
	return one_hot


# ========================
# Функция потерь
# ========================

def cross_entropy_loss(probs, y_true):
	if len(probs) == 0 or len(y_true) == 0:
		return 0.0
	B = len(probs)
	loss = 0.0
	for i in range(B):
		for j in range(len(probs[i])):
			p = probs[i][j]
			if p < 0.1 ** 15:
				p = 0.1 ** 15
			loss += y_true[i][j] * log(p)
	return -loss / B


# ========================
# Обратное распространение
# ========================

def backward_pass(probs, y_true_one_hot, context):
	B = len(probs)
	num_layers = len(context['network']['layers'])
	
	# Градиент на выходе: dZ = probs - y_true
	dZ_out = []
	for i in range(B):
		row = []
		for j in range(len(probs[i])):
			row.append(probs[i][j] - y_true_one_hot[i][j])
		dZ_out.append(row)
	
	# Градиенты выходного слоя
	A_prev = context['activations'][-1]
	dW_out = matmul(transpose(dZ_out), A_prev)
	db_out = []
	for j in range(len(dZ_out[0])):
		db_out.append(0.0)
	for i in range(B):
		for j in range(len(dZ_out[0])):
			db_out[j] += dZ_out[i][j]
	for j in range(len(db_out)):
		db_out[j] /= B
	for i in range(len(dW_out)):
		for j in range(len(dW_out[0])):
			dW_out[i][j] /= B

	# Распространение назад
	dA = matmul(dZ_out, context['network']['output_layer']['W'])

	layer_grads = []
	for layer_idx in range(num_layers - 1, -1, -1):
		Z = context['z_values'][layer_idx]
		W = context['network']['layers'][layer_idx]['W']
		A_prev = context['activations'][layer_idx]

		if context['activation_function'] == 'sigmoid':
			activation_deriv = sigmoid_derivative_matrix(Z)
		elif context['activation_function'] == 'relu':
			activation_deriv = relu_derivative_matrix(Z)
		else:
			activation_deriv = relu_derivative_matrix(Z)

		dZ = []
		for i in range(len(dA)):
			row = []
			for j in range(len(dA[i])):
				row.append(dA[i][j] * activation_deriv[i][j])
			dZ.append(row)

		dW = matmul(transpose(dZ), A_prev)
		db = []
		for j in range(len(dZ[0])):
			db.append(0.0)
		for i in range(len(dZ)):
			for j in range(len(dZ[0])):
				db[j] += dZ[i][j]
		for j in range(len(db)):
			db[j] /= B
		for i in range(len(dW)):
			for j in range(len(dW[0])):
				dW[i][j] /= B

		layer_grads.insert(0, {'dW': dW, 'db': db})

		if layer_idx > 0:
			dA = matmul(dZ, W)

	return layer_grads, {'dW': dW_out, 'db': db_out}


# ========================
# Обновление весов
# ========================

def update_weights(layers, output_layer, layer_grads, output_grad, learning_rate):
	for i in range(len(layers)):
		dW = layer_grads[i]['dW']
		db = layer_grads[i]['db']
		W = layers[i]['W']
		b = layers[i]['b']
		for r in range(len(W)):
			for c in range(len(W[0])):
				W[r][c] -= learning_rate * dW[r][c]
		for j in range(len(b)):
			b[j] -= learning_rate * db[j]

	dW_out = output_grad['dW']
	db_out = output_grad['db']
	W_out = output_layer['W']
	b_out = output_layer['b']
	for r in range(len(W_out)):
		for c in range(len(W_out[0])):
			W_out[r][c] -= learning_rate * dW_out[r][c]
	for j in range(len(b_out)):
		b_out[j] -= learning_rate * db_out[j]


# ========================
# Разбиение на батчи
# ========================


def create_batches(X, y, batch_size):
	batches = []
	n_samples = len(X)
	
	for start_idx in range(0, n_samples, batch_size):
		end_idx = start_idx + batch_size
		if end_idx > n_samples:
			end_idx = n_samples
		
		X_batch = []
		y_batch = []
		for i in range(start_idx, end_idx):
			X_batch.append(X[i])
			y_batch.append(y[i])
		
		batches.append((X_batch, y_batch))
	
	return batches


# ========================
# Обучение
# ========================

def train_network(
	X_train, y_train,
	hidden_layers_count=2,
	hidden_size=5,
	output_size=2,
	epochs=100,
	lr=0.1,
	batch_size=32, 
	activation_function='relu'
):
	_, input_size = shape(X_train)
	network = init_network(input_size, hidden_layers_count, hidden_size, output_size)

	for epoch in range(epochs):
		# Создаём батчи
		batches = create_batches(X_train, y_train, batch_size)
		epoch_loss = 0.0
		num_batches = len(batches)

		for X_batch, y_batch in batches:
			# Преобразуем метки в one-hot
			y_batch_one_hot = to_one_hot(y_batch, output_size)
			
			# Прямой проход
			probs, context = forward_pass(X_batch, network, activation_function)
			if probs == None:
				print("Ошибка в прямом проходе")
				continue
			
			# Потеря
			loss = cross_entropy_loss(probs, y_batch_one_hot)
			epoch_loss += loss
			
			# Обратный проход
			layer_grads, output_grad = backward_pass(probs, y_batch_one_hot, context)
			
			# Обновление весов
			update_weights(
				network['layers'],
				network['output_layer'],
				layer_grads,
				output_grad,
				lr
			)
		
		# Средняя потеря за эпоху
		avg_loss = epoch_loss / num_batches
		if epoch % 1 == 0:
			print("Epoch ", epoch, "Loss: ", avg_loss)
	
	return network


def train_best_loss_network(
	X_train, y_train,
	hidden_layers_count=2,
	hidden_size=5,
	output_size=2,
	epochs=100,
	lr=0.1,
	batch_size=32, 
	activation_function='relu',
	X_val=None,
	y_val=None,
	pretrained_network=None
):
	if pretrained_network != None:
		network = deep_copy_network(pretrained_network)
		if len(network['layers']) > 0:
			input_size = len(network['layers'][0]['W'][0])
		else:
			input_size = len(network['output_layer']['W'][0])
	else:
		_, input_size = shape(X_train)
		network = init_network(input_size, hidden_layers_count, hidden_size, output_size)
	
	# Используем валидацию, если есть, иначе обучение
	use_validation = (X_val != None) and (y_val != None)
	
	best_loss = 999999999
	best_network = None

	for epoch in range(epochs):
		# Обучение по батчам
		batches = create_batches(X_train, y_train, batch_size)
		train_loss = 0.0
		num_batches = len(batches)

		for X_batch, y_batch in batches:
			y_batch_one_hot = to_one_hot(y_batch, output_size)
			probs, context = forward_pass(X_batch, network, activation_function)
			if probs == None:
				continue
			loss = cross_entropy_loss(probs, y_batch_one_hot)
			train_loss += loss
			
			layer_grads, output_grad = backward_pass(probs, y_batch_one_hot, context)
			update_weights(
				network['layers'],
				network['output_layer'],
				layer_grads,
				output_grad,
				lr
			)
		
		avg_train_loss = train_loss / num_batches
		
		# Вычисляем loss для оценки качества
		if use_validation:
			# Валидационный loss
			probs_val, _ = forward_pass(X_val, network, activation_function)
			if probs_val == None:
				val_loss = 999999999
			else:
				y_val_one_hot = to_one_hot(y_val, output_size)
				val_loss = cross_entropy_loss(probs_val, y_val_one_hot)
			current_loss = val_loss
		else:
			# Используем обучающий loss
			current_loss = avg_train_loss

		# Сохраняем лучшую сеть
		if current_loss < best_loss:
			best_loss = current_loss
			best_network = deep_copy_network(network)
			# print(f"Новый рекорд на эпохе {epoch}: loss = {best_loss:.4f}")

		if epoch % 1 == 0:
			if use_validation:
				print("Epoch ", epoch, "Train Loss: ", avg_train_loss, "Val Loss: ", val_loss)
			else:
				print("Epoch ", epoch, "Train Loss: ", avg_train_loss)

	return best_network  # ← возвращаем лучшую сеть, а не последнюю!


# ========================
# Предсказание и оценка
# ========================

def predict(X, network, activation_function):
	probs, _ = forward_pass(X, network, activation_function)
	return probs


def generate_dataset(n_samples=100):
	# Генерирует синтетический датасет для бинарной классификации.
	# Признаки: [рост (см), вес (кг), возраст (лет)]
	# Метка: 0 = женщина, 1 = мужчина

	# Правило:
	#   - Если рост < 170 ИЛИ вес < 65 → класс 0
	#   - Иначе → класс 1
	
	X = []
	y = []
	
	for _ in range(n_samples):
		height = 150 + int(40 * random())
		weight = 45 + int(55 * random())
		age = 18 + int(42 * random())
		
		if height < 170 or weight < 65:
			label = 0
		else:
			label = 1
		
		X.append([height, weight, age])
		y.append(label)
	
	return X, y


def compute_mean_std(X):
	n_samples, n_features = shape(X)
	if n_samples == 0:
		return [], []
	
	# Среднее
	mean = []
	for j in range(n_features):
		total = 0.0
		for i in range(n_samples):
			total += X[i][j]
		mean.append(total / n_samples)

	std = []
	for j in range(n_features):
		total_sq = 0.0
		for i in range(n_samples):
			diff = X[i][j] - mean[j]
			total_sq += diff * diff
		variance = total_sq / n_samples

		if variance < 0.1 ** 8:
			std.append(1.0)
		else:
			std.append(variance ** 0.5)
	
	return mean, std


def normalize(X, mean, std):
	n_samples, n_features = shape(X)
	
	X_norm = []
	for i in range(n_samples):
		row = []
		for j in range(n_features):
			normalized_val = (X[i][j] - mean[j]) / std[j]
			row.append(normalized_val)
		X_norm.append(row)
	return X_norm


def get_columns(data, start_col, end_col):
	result = []
	for row in data:
		new_row = []
		for j in range(start_col, end_col):
			new_row.append(row[j])
		result.append(new_row)
	return result


def train_test_split(X, y, test_size=0.2):
	if len(X) != len(y):
		print("Error: X and y must have the same length")
		return None, None, None, None
	
	n_samples = len(X)
	if n_samples == 0:
		return [], [], [], []
	
	if test_size <= 0 or test_size >= 1:
		print("Error: test_size must be between 0 and 1")
		return None, None, None, None

	n_test = int(n_samples * test_size)
	n_train = n_samples - n_test

	if n_train == 0:
		print("Error: training set is empty (test_size too large)")
		return None, None, None, None

	X_train = []
	y_train = []
	X_test = []
	y_test = []

	for i in range(n_train):
		X_train.append(X[i])
		y_train.append(y[i])

	for i in range(n_train, n_samples):
		X_test.append(X[i])
		y_test.append(y[i])

	return X_train, X_test, y_train, y_test


def deep_copy_network(network):
	### Создаёт глубокую копию сети (весов и смещений)
	layers_copy = []
	for layer in network['layers']:
		W_copy = []
		for row in layer['W']:
			row_copy = []
			for val in row:
				row_copy.append(val)
			W_copy.append(row_copy)
		b_copy = []
		for val in layer['b']:
			b_copy.append(val)
		layers_copy.append({'W': W_copy, 'b': b_copy})

	W_out_copy = []
	for row in network['output_layer']['W']:
		row_copy = []
		for val in row:
			row_copy.append(val)
		W_out_copy.append(row_copy)
	b_out_copy = []
	for val in network['output_layer']['b']:
		b_out_copy.append(val)
	
	return {
		'layers': layers_copy,
		'output_layer': {'W': W_out_copy, 'b': b_out_copy}
	}

