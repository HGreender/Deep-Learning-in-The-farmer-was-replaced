E = 2.718281828459045

ALGORITHMS_LIST = [
	'snake',
	'sequence',
	'mega_pumpkin',
	'sequence_row',
	'async_mega_pumpkin',
	'async_snake_mega_pumpkin'
]


WORLD_SIZE = get_world_size()
MAX_DRONES = max_drones() 

X_DIRS = West, East
Y_DIRS = South, North 


SOIL_PLANTS = [
	Entities.Sunflower,
	Entities.Carrot,
	Entities.Pumpkin,
	Entities.Cactus,
]

PLANTS_LIST = [
	Entities.Sunflower,
	Entities.Grass,
	Entities.Bush,
	Entities.Carrot,
	Entities.Tree,
	Entities.Pumpkin,
	Entities.Cactus
]

def get_full_entities_list():
	### Returns a list with all the entities in the game
	FULL_ENTITIES_LIST = list()
	for entity in Entities:
		FULL_ENTITIES_LIST.append(entity)
	return FULL_ENTITIES_LIST

FULL_ENTITIES_LIST = get_full_entities_list()