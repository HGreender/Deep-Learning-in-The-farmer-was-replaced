from Constants import E


def exp(x):
	return E ** x


def sqrt(x):
	return x ** (1/2)


def uniform_random(low, high):
	return low + (high - low) * random()


def transpose(M):
	if len(M) == 0:
		return []
	rows = len(M)
	cols = len(M[0])
	MT = []
	for j in range(cols):
		new_row = []
		for i in range(rows):
			new_row.append(M[i][j])
		MT.append(new_row)
	return MT


def matmul(A, B):
	### Multiplies matrix A with Matrix B
	### >>> Returns the resulting matrix C
	rows_A = len(A)
	if rows_A > 0:
		cols_A = len(A[0])
	else:
		cols_A = 0
	rows_B = len(B)
	if rows_B > 0:
		cols_B = len(B[0])
	else:
		cols_B = 0

	if cols_A != rows_B:
		print('ValueError("Число столбцов первой матрицы должно быть равно числу строк второй.")')
		return

	C = list()
	for _ in range(rows_A):
		temp = list()
		for _ in range(cols_B):
			temp.append(0)
		C.append(temp)

	for i in range(rows_A):
		for j in range(cols_B):
			for k in range(cols_A):
				C[i][j] += A[i][k] * B[k][j]

	return C


def create_random_array(m, n):
	if (n <= 0) or (m <= 0):
		print('RandomArrayError: (m, n) must be more than zero')
		return
	
	W = list()
	limit = sqrt(6.0 / n)
	for i in range(m):
		row = []
		for j in range(n):
			row.append(uniform_random(-limit, limit))
		W.append(row)
	return W


def add_matrices(A, B):
	rows_A = len(A)
	if rows_A == 0:
		if len(B) == 0:
			return []
		else:
			print("ShapeError: матрицы имеют разное число строк")
			return None

	cols_A = len(A[0])
	rows_B = len(B)
	if rows_B == 0:
		print("ShapeError: матрицы имеют разное число строк")
		return None

	cols_B = len(B[0])

	if rows_A != rows_B or cols_A != cols_B:
		print("ShapeError: размеры матриц не совпадают")
		return None

	C = []
	for i in range(rows_A):
		new_row = []
		for j in range(cols_A):
			new_row.append(A[i][j] + B[i][j])
		C.append(new_row)

	return C


def softmax(z):
	if len(z) == 0:
		return []
	
	max_val = z[0]
	for i in range(1, len(z)):
		max_val = max(z[i], max_val)
	
	exp_vals = []
	for i in range(len(z)):
		exp_vals.append(exp(z[i] - max_val))
	
	sum_exp = 0.0
	for val in exp_vals:
		sum_exp += val
	
	probs = []
	for val in exp_vals:
		probs.append(val / sum_exp)
	
	return probs


def add_bias_matrix(Z, b):
	### Добавляет вектор b к каждой строке матрицы Z
	if len(Z) == 0:
		return Z
	n_out = len(b)
	B = len(Z)

	for i in range(B):
		if len(Z[i]) != n_out:
			print("BiasError: несовместимая длина строки и bias")
			return Z

	for i in range(B):
		for j in range(n_out):
			Z[i][j] += b[j]
	return Z


def softmax_batch(logits):
	### Применяет softmax к каждой строке матрицы logits
	probs = []
	for i in range(len(logits)):
		row = logits[i]
		probs.append(softmax(row))
	return probs


def sigmoid(x):
	### Сигмоида для одного числа
	ex = exp(-x)
	return 1.0 / (1.0 + ex)


def sigmoid_vector(v):
	### Применяет сигмоиду к каждому элементу вектора
	result = []
	for val in v:
		result.append(sigmoid(val))
	return result


def sigmoid_matrix(M):
	### Применяет сигмоиду к каждой ячейке матрицы
	result = []
	for row in M:
		result.append(sigmoid_vector(row))
	return result


def sigmoid_derivative(x):
	s = sigmoid(x)
	return s * (1.0 - s)


def sigmoid_derivative_vector(v):
	result = []
	for val in v:
		result.append(sigmoid_derivative(val))
	return result


def sigmoid_derivative_matrix(M):
	result = []
	for row in M:
		result.append(sigmoid_derivative_vector(row))
	return result


def log(x, base=None, tol=0.1**12, max_iter=1000):
	# Параметры:
	# 	x (float): положительное число, для которого вычисляется логарифм.
	# 	base (float or None): основание логарифма. Если None — натуральный логарифм (ln).
	# 	tol (float): точность вычисления.
	# 	max_iter (int): максимальное число итераций.
	
	# Возвращает:
	# 	float: приближённое значение логарифма.
	if x <= 0:
		print('ValueError("Аргумент логарифма должен быть положительным.")')
		return None
	
	if base != None:
		if base <= 0 or base == 1:
			print('ValueError("Основание логарифма должно быть положительным и не равным 1.")')
			return 
		return log(x) / log(base)

	if x == 1.0:
		return 0.0

	k = 0
	y = x
	ln2 = 0.6931471805599453

	while y < 0.5:
		y *= 2
		k -= 1
	while y > 1.5:
		y /= 2
		k += 1

	z = y - 1.0

	term = z
	ln_y = term
	n = 1
	while abs(term) > tol and n < max_iter:
		n += 1
		term *= -z * (n - 1) / n
		ln_y += term

	return ln_y + k * ln2


def int(x):
	if x >= 0:
		return x // 1
	else:
		return -((-x) // 1)


def relu(x):
	# ReLU активация: f(x) = max(0, x)
	if x > 0:
		return x
	else:
		return 0.0


def relu_vector(v):
	# Применяет ReLU к каждому элементу вектора
	result = []
	for val in v:
		result.append(relu(val))
	return result


def relu_matrix(M):
	# Применяет ReLU к каждой ячейке матрицы
	result = []
	for row in M:
		result.append(relu_vector(row))
	return result


def relu_derivative(x):
	# Производная ReLU: 1 если x > 0, иначе 0
	if x > 0:
		return 1.0
	else:
		return 0.0


def relu_derivative_vector(v):
	# Производная ReLU по вектору
	result = []
	for val in v:
		result.append(relu_derivative(val))
	return result


def relu_derivative_matrix(M):
	# Производная ReLU по матрице
	result = []
	for row in M:
		result.append(relu_derivative_vector(row))
	return result