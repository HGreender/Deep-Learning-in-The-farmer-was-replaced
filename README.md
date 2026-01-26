# Deep Learning in [The farmer was replaced](https://store.steampowered.com/app/2060160/The_Farmer_Was_Replaced/)
A library for implementing a FCNN (fully connected neural network)

NeuroNet — Нейросеть с нуля для игры ["The farmer was replaced"](https://store.steampowered.com/app/2060160/The_Farmer_Was_Replaced/)

## ОПИСАНИЕ
Этот проект реализует полностью функциональную нейронную сеть на чистом Python без использования NumPy, TensorFlow или PyTorch, так как игра их не поддерживает.
Все операции (умножение матриц, активации, обратное распространение) написаны вручную с использованием только циклов и базовых арифметических операций.

## ОСОБЕННОСТИ
+ Полносвязная нейронная сеть (MLP)
+ Поддержка ReLU и сигмоиды
+ Softmax и кросс-энтропия для классификации
+ Обучение по батчам
+ Нормализация данных
+ Сохранение лучшей модели по loss
+ Дообучение предварительно обученной сети
+ Работает с числовыми данными и строковыми метками (автоматическое преобразование)

## ФУНКЦИИ
Основные функции:

+ train_best_loss_network() — обучение сети с сохранением лучшей модели или дообучение с загрузкой предобученных весов
+ predict() — предсказание на новых данных
+ deep_copy_network() — глубокое копирование сети
+ normalize() — нормализация данных (поддерживает как батчи, так и одиночные объекты)

### Подготовка данных:

+ train_test_split() — разделение на обучающую и тестовую выборки
+ to_one_hot() — преобразование меток в one-hot кодировку
+ compute_mean_std(), normalize() — стандартизация признаков

### Гиперпараметры:

+ hidden_layers_count — количество скрытых слоёв
+ hidden_size — число нейронов в скрытом слое
+ output_size — число классов
+ epochs — количество эпох обучения
+ lr — скорость обучения
+ batch_size — размер батча
+ activation_function — 'relu' или 'sigmoid'

## ИСПОЛЬЗОВАНИЕ

#### 1. Подготовьте данные в формате:

X = [[признак1, признак2, ...], ...]
y = [метка1, метка2, ...] # целые числа от 0 до (число_классов - 1)

#### 2. Разделите данные:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#### 3. Нормализуйте:

mean, std = compute_mean_std(X_train)
X_train_norm = normalize(X_train, mean, std)
X_test_norm = normalize(X_test, mean, std)

#### 4. Обучите модель:

model = train_best_loss_network(
X_train_norm, y_train,
hidden_layers_count=2,
hidden_size=16,
output_size=2,
epochs=300,
lr=0.1,
batch_size=8,
activation_function='relu'
)

#### 5. Оцените качество:

probs = predict(X_test_norm, model)
Преобразуйте вероятности в предсказания

## ДООБУЧЕНИЕ

Можно дообучить уже существующую модель:
updated_model = train_best_loss_network(
X_new, y_new,
epochs=50,
lr=0.01,
pretrained_network=model
)

## ГОТОВЫЕ ПРИМЕРЫ
Проект включает пример датасета h_w_f_g — данные о росте, весе, размере стопы и поле. Автоматически преобразуется в числовые признаки (ИМТ, возраст) и метки (0/1).

А также: 

+ Уже обученные веса saved_weght_acc_66 в weights.py с печальной Accuracy = 66% (вы можете это исправить! - в этом вся суть игры),
+ NeuroTrain.py для теста обучения,
+ NeuroPredict.py для тестовых предсказаний на готовой моделе,
+ NeuroAccuracy.py - для оценки точности модели,
+ NeuroFineTune.py - для дообучения модели на других данных или параметрах

