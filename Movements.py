from Constants import SOIL_PLANTS, WORLD_SIZE, X_DIRS, Y_DIRS


LIB_ERROR = 'ERROR in Movements'


def get_coordinates():
	### Returns the current coordinates
	return get_pos_x(), get_pos_y()


def plant_(my_plant):
	### Checks the soil and plants the [my_plant]
	if my_plant in SOIL_PLANTS:
		if get_ground_type() == Grounds.Grassland:
			till()
		plant(my_plant)
	else:
		if get_ground_type() == Grounds.Soil:
			till()
		plant(my_plant)


def plant_by_coords(my_plant, coords_list):
	### Plants the [my_plant] by [coords_list]
	for x, y in coords_list:
		move_to(x, y)
		plant_(my_plant)


def multi_move(n, dir):
	### Moving by [n] cells in the [dir] direction
	### >>> Returns the resulting coordinates
	for i in range(n):
		move(dir)
	return get_coordinates()


def move_like_snake(i, j):
	### Snake movement for each iteration while iterating through the entire field
	world_last_value = WORLD_SIZE - 1

	if i % 2 == 0:
		if j != world_last_value:
			move(East)
		else:
			move(South)
	else:
		if j != world_last_value:
			move(West)
		else:
			move(South)


def move_to_row_start():
	### Move the drone to start coordinates	
	### >>> Returns the resulting coordinates
	x = 0
	y = get_pos_y()
	move_to(x, y)
	return x, y


def move_to(x1, y1):
	### Moves the drone to the coordinates sent to the function
	### >>> Returns the resulting coordinates
	func_name = 'move_to():'

	x0 = get_pos_x()
	y0 = get_pos_y()

	if x1 > WORLD_SIZE or y1 > WORLD_SIZE:
		print(LIB_ERROR, func_name, 'ValueError(Переданные координаты за пределами поля)')
		return None, None

	out_offset = WORLD_SIZE - abs(x1 - x0)
	in_offse = abs(x1 - x0)
	min_offset = min(out_offset, in_offse)

	# False = 0 (West), True = 1 (East)
	is_East1 = min_offset == out_offset and x0 > x1
	is_East2 = min_offset == in_offse and x1 > x0
	x_dir = X_DIRS[is_East1 or is_East2]

	multi_move(min_offset, x_dir)

	out_offset = WORLD_SIZE - abs(y1 - y0)
	in_offse = abs(y1 - y0)
	min_offset = min(out_offset, in_offse)

	# False = 0 (South), True = 1 (North)
	is_North1 = min_offset == out_offset and y0 > y1
	is_North2 = min_offset == in_offse and y1 > y0
	y_dir = Y_DIRS[is_North1 or is_North2]

	x, y = multi_move(min_offset, y_dir)

	return x, y


def move_to_start():
	### Move the drone to start coordinates	
	### !!! CAUTION !!! I'm using the upper-left corner as the starting coordinates.
	world_last_value = WORLD_SIZE - 1
	move_to(0, world_last_value)
