_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world', 'yolo_world.hooks'],
                      allow_failed_imports=False)

METAINFO = {
        'classes':
        ('aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock',
         'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet',
         'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
         'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
         'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
         'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
         'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
         'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
         'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
         'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
         'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
         'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
         'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
         'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
         'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
         'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
         'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
         'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
         'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
         'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
         'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
         'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
         'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
         'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
         'bottle_opener', 'bouquet', 'bow_(weapon)',
         'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl',
         'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders',
         'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread',
         'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach',
         'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket',
         'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train',
         'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed',
         'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter',
         'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet',
         'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder',
         'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can',
         'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane',
         'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen',
         'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
         'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
         'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
         'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
         'cash_register', 'casserole', 'cassette', 'cast', 'cat',
         'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery',
         'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue',
         'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard',
         'cherry', 'chessboard', 'chicken_(animal)', 'chickpea',
         'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)',
         'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk',
         'chocolate_mousse', 'choker', 'chopping_board', 'chopstick',
         'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette',
         'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent',
         'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard',
         'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower',
         'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat',
         'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)',
         'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil',
         'coin', 'colander', 'coleslaw', 'coloring_material',
         'combination_lock', 'pacifier', 'comic_book', 'compass',
         'computer_keyboard', 'condiment', 'cone', 'control',
         'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
         'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
         'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
         'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
         'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
         'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
         'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
         'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
         'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
         'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
         'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
         'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
         'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table',
         'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
         'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
         'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
         'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove',
         'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat',
         'dress_suit', 'dresser', 'drill', 'drone', 'dropper',
         'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling',
         'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle',
         'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg',
         'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair',
         'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot',
         'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret',
         'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine',
         'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine',
         'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug',
         'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod',
         'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash',
         'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)',
         'flower_arrangement', 'flute_glass', 'foal', 'folding_chair',
         'food_processor', 'football_(American)', 'football_helmet',
         'footstool', 'fork', 'forklift', 'freight_car', 'French_toast',
         'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge',
         'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose',
         'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin',
         'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger',
         'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove',
         'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart',
         'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater',
         'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle',
         'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun',
         'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger',
         'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass',
         'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle',
         'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil',
         'headband', 'headboard', 'headlight', 'headscarf', 'headset',
         'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
         'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
         'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
         'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
         'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
         'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
         'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
         'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
         'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
         'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
         'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
         'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
         'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
         'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
         'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
         'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade',
         'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
         'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
         'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
         'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
         'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange',
         'manger', 'manhole', 'map', 'marker', 'martini', 'mascot',
         'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)',
         'matchbox', 'mattress', 'measuring_cup', 'measuring_stick',
         'meatball', 'medicine', 'melon', 'microphone', 'microscope',
         'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake',
         'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)',
         'money', 'monitor_(computer_equipment) computer_monitor', 'monkey',
         'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle',
         'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad',
         'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument',
         'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle',
         'nest', 'newspaper', 'newsstand', 'nightshirt',
         'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook',
         'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)',
         'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion',
         'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven',
         'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle',
         'padlock', 'paintbrush', 'painting', 'pajamas', 'palette',
         'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose',
         'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
         'paperweight', 'parachute', 'parakeet', 'parasail_(sports)',
         'parasol', 'parchment', 'parka', 'parking_meter', 'parrot',
         'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
         'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
         'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
         'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
         'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
         'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
         'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
         'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
         'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
         'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
         'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
         'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
         'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
         'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
         'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot',
         'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn',
         'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller',
         'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin',
         'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt',
         'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver',
         'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry',
         'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
         'recliner', 'record_player', 'reflector', 'remote_control',
         'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
         'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
         'rolling_pin', 'root_beer', 'router_(computer_equipment)',
         'rubber_band', 'runner_(carpet)', 'plastic_bag',
         'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
         'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
         'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
         'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
         'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
         'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
         'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
         'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
         'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
         'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
         'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
         'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
         'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
         'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
         'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
         'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
         'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
         'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
         'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
         'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
         'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
         'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
         'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
         'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew',
         'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove',
         'strainer', 'strap', 'straw_(for_drinking)', 'strawberry',
         'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer',
         'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower',
         'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants',
         'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit',
         'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
         'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
         'tambourine', 'army_tank', 'tank_(storage_vessel)',
         'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
         'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
         'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
         'telephone_pole', 'telephoto_lens', 'television_camera',
         'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
         'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
         'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer',
         'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster',
         'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs',
         'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover',
         'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy',
         'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike',
         'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray',
         'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod',
         'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban',
         'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
         'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
         'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
         'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
         'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
         'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
         'washbasin', 'automatic_washer', 'watch', 'water_bottle',
         'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
         'water_gun', 'water_scooter', 'water_ski', 'water_tower',
         'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
         'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
         'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
         'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
         'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
         'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
         'yoke_(animal_equipment)', 'zebra', 'zucchini'),
        'palette':
        None
    }

# hyper-parameters
num_classes = 1203
num_training_classes = 80
#max_epochs = 100  # Maximum training epochs
max_epochs = 5
#close_mosaic_epochs = 2
close_mosaic_epochs = 5
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
#base_lr = 2e-3
base_lr = 2e-4
#base_lr = 2e-5
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 8
# text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
text_model_name = 'openai/clip-vit-base-patch32'
# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

vps_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True), # 保持和训练一致
    dict(type='Pad', size=(640, 640), pad_val=dict(img=114)),
    dict(type='LoadText'),
    dict(type='LoadVisualMask', scale_factor=1/8),          # 生成 Mask
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts', 'visual_masks'))        # 必须包含 visual_masks
]

# [新增] 定义 VPS Dataloader 配置 (传给 Hook 用)
lvis_vps_dataloader = dict(
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root='../../datasets/lvis_train_vps/',
            ann_file='lvis_train_vps.json',
            metainfo=METAINFO,
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)
        ),
        class_text_path='data/texts/lvis_v1_class_texts.json',
        pipeline=vps_pipeline
    )
)

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='LoadVisualMask'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts', 'visual_masks'))
]
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]

obj365v1_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',
        data_root='../../datasets/Objects365v1',
        ann_file='annotations/objects365_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/obj365v1_class_texts.json',
    pipeline=train_pipeline)

mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
                        data_root='../../../dataset/GQA/',
                        ann_file='final_mixed_train_no_coco.json',
                        data_prefix=dict(img='images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=train_pipeline)

flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='../../../dataset/flickr 30k/',
    ann_file='final_flickr_separateGT_train.json',
    data_prefix=dict(img='flickr30k-images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)

# obj365v1_train_dataset = dict(
#     type='MultiModalDataset',
#     dataset=dict(
#         type='YOLOv5Objects365V1Dataset',
#         data_root='../../datasets/Objects365v1/object365v1_sampled/sampling_ratio_0.001',
#         ann_file='annotations/objects365_train_sampled_0.10%.json',
#         data_prefix=dict(img='images/'),
#         filter_cfg=dict(filter_empty_gt=False, min_size=32)),
#     class_text_path='data/texts/obj365v1_class_texts.json',
#     pipeline=train_pipeline)
#
# mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
#                         data_root='../../../dataset/GQA/gqa_sampled/sampling_ratio_0.001/',
#                         ann_file='annotations/final_mixed_train_no_coco_sampled_0.10%.json',
#                         data_prefix=dict(img='images/'),
#                         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#                         pipeline=train_pipeline)
#
# flickr_train_dataset = dict(
#     type='YOLOv5MixedGroundingDataset',
#     data_root='../../../dataset/flickr 30k/flickr_sampled/sampling_ratio_0.001/',
#     ann_file='annotations/final_flickr_separateGT_train_sampled_0.10%.json',
#     data_prefix=dict(img='images/'),
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         obj365v1_train_dataset,
                                         flickr_train_dataset, mg_train_dataset
                                     ],
                                     ignore_keys=['classes', 'palette']))

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='../../../dataset/coco/',
                 test_mode=True,
                 ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
                 data_prefix=dict(img='images'),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='../../../dataset/coco/annotations/lvis_v1_minival_inserted_image_name.json',
                     metric='bbox')
test_evaluator = val_evaluator

# coco_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type='YOLOv5LVISV1Dataset',
#                  data_root='../../../dataset/coco/',
#                  test_mode=True,
#                  ann_file='annotations/lvis_v1_val.json',
#                  data_prefix=dict(img='images/'),
#                  batch_shapes_cfg=None),
#     class_text_path='data/texts/lvis_v1_class_texts.json',
#     pipeline=test_pipeline)
# val_dataloader = dict(dataset=coco_val_dataset, #下面两项是debug的时候需要的,正式训练删掉
#                       # num_workers=0,
#                       # persistent_workers=False
#                       )
# test_dataloader = val_dataloader
#
# val_evaluator = dict(type='mmdet.LVISMetric',
#                      ann_file='../../../dataset/coco/annotations/lvis_v1_val.json',
#                      metric='bbox')
# test_evaluator = val_evaluator

# training settings
# default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
#                      checkpoint=dict(interval=1,
#                                      rule='greater'))
# 定义具体的调度策略：全程保持学习率不变
# param_scheduler = [
#     dict(
#         type='ConstantLR',
#         factor=1.0,       # 1.0 表示始终保持 base_lr 的 100% (即 2e-5)
#         by_epoch=True,    # 按 epoch 更新
#         begin=0,          # 从第 0 epoch 开始
#         end=5             # 到第 5 epoch 结束 (对应你的 max_epochs=5)
#     )
# ]
# default_hooks = dict(
#     param_scheduler=dict(
#         _delete_=True,            # <--- 必须加！清除父配置的参数
#         type='ParamSchedulerHook' # 换成 MMEngine 的标准调度器钩子
#     ),
#     checkpoint=dict(interval=1, rule='greater')
# )

# [推荐配置] 适用于短周期(5-20 epoch)的新模块对齐训练
param_scheduler = [
    # 1. Warmup 阶段: 第 0-1 个 epoch，学习率从 0.001*lr 线性增加到 lr
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=1,  # 热身 1 个 epoch
        convert_to_iter_based=True  # 自动转换为基于 iter 的平滑热身
    ),
    # 2. Constant 阶段: 第 1-5 个 epoch，保持全速学习率
    dict(
        type='ConstantLR',
        factor=1.0,
        by_epoch=True,
        begin=1,
        end=max_epochs  # 直到训练结束
    )
]

# 必须更新 default_hooks 以应用新的调度器
default_hooks = dict(
    param_scheduler=dict(
        _delete_=True,            # 清除 yolov8 默认的调度器
        type='ParamSchedulerHook' # 使用 MMEngine 通用调度器
    ),
    checkpoint=dict(interval=1, rule='greater') # 每个 epoch 保存
)

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2),
    dict(type='VisualPromptInjectionHook',
         dataloader_cfg=lvis_vps_dataloader, # 传入上面的配置
         num_classes=1203,                   # LVIS 类别数
         priority='NORMAL')
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=1,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
# optim_wrapper = dict(optimizer=dict(
#     _delete_=True,
#     type='AdamW',
#     lr=base_lr,
#     weight_decay=weight_decay,
#     batch_size_per_gpu=train_batch_size_per_gpu),
#                      paramwise_cfg=dict(bias_decay_mult=0.0,
#                                         norm_decay_mult=0.0,
#                                         custom_keys={
#                                             'backbone.text_model':
#                                             dict(lr_mult=0.01),
#                                             'logit_scale':
#                                             dict(weight_decay=0.0)
#                                         }),
#                      constructor='YOLOWv5OptimizerConstructor')


#只训练图像提示模块和正交分解模块
# optim_wrapper = dict(
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=base_lr,  # 这里的 base_lr 是 2e-3
#         weight_decay=weight_decay,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#
#     # [核心修改] 参数级精细化配置
#     paramwise_cfg=dict(
#         bias_decay_mult=0.0,
#         norm_decay_mult=0.0,
#
#         # custom_keys 字典：键是参数名的一部分，值是配置
#         custom_keys={
#             # 1. 冻结 Backbone (包含 Image Encoder)
#             'backbone': dict(lr_mult=0.0),
#
#             # 2. 冻结 Text Encoder (CLIP)
#             'text_model': dict(lr_mult=0.0),
#
#             # 3. 冻结 Neck (PAFPN)
#             'neck': dict(lr_mult=0.0),
#
#             # 4. 冻结 Logit Scale (温度系数)
#             'logit_scale': dict(lr_mult=0.0),
#
#             # 5. [关键] 冻结整个 Head
#             'bbox_head': dict(lr_mult=0.0),
#
#             # 6. [核心] 唯独解冻 SAVPE
#             # 这里的 key 必须写得比 'bbox_head' 更长、更具体，以触发最长匹配原则
#             # 参数的全名通常是: bbox_head.head_module.savpe.xxx
#             'bbox_head.head_module.savpe': dict(lr_mult=1.0, decay_mult=1.0),
#             'bbox_head.head_module.opr_fusion': dict(lr_mult=1.0, decay_mult=1.0),
#         }
#     ),
#     constructor='YOLOWv5OptimizerConstructor'
# )

#下面这个优化器是模块和head头（学习率乘0.1）一起训练
# optim_wrapper = dict(
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=base_lr,  # 这里的 base_lr 是 2e-3
#         weight_decay=weight_decay,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#
#     # [核心修改] 参数级精细化配置
#     paramwise_cfg=dict(
#         bias_decay_mult=0.0,
#         norm_decay_mult=0.0,
#
#         # custom_keys 字典：键是参数名的一部分，值是配置
#         custom_keys={
#             # 1. 冻结 Backbone (包含 Image Encoder)
#             'backbone': dict(lr_mult=0.0),
#
#             # 2. 冻结 Text Encoder (CLIP)
#             'text_model': dict(lr_mult=0.0),
#
#             # 3. 冻结 Neck (PAFPN)
#             'neck': dict(lr_mult=0.0),
#
#             # 4. 冻结 Logit Scale (温度系数)
#             'logit_scale': dict(lr_mult=0.0),
#
#             # 5. Head 部分学习率乘以0.1
#             'bbox_head': dict(lr_mult=0.1),
#
#             # 6. [核心] 唯独解冻 SAVPE
#             # 这里的 key 必须写得比 'bbox_head' 更长、更具体，以触发最长匹配原则
#             # 参数的全名通常是: bbox_head.head_module.savpe.xxx
#             'bbox_head.head_module.savpe': dict(lr_mult=1.0, decay_mult=1.0),
#             'bbox_head.head_module.opr_fusion': dict(lr_mult=1.0, decay_mult=1.0),
#         }
#     ),
#     constructor='YOLOWv5OptimizerConstructor'
# )

#只训练deformablepromptencoder，但是同时有对其损失
# optim_wrapper = dict(
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=weight_decay,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#
#     # [核心修改] 参数级精细化配置
#     paramwise_cfg=dict(
#         bias_decay_mult=0.0,
#         norm_decay_mult=0.0,
#
#         # custom_keys 字典：键是参数名的一部分，值是配置
#         custom_keys={
#             # 1. 冻结 Backbone (包含 Image Encoder)
#             'backbone': dict(lr_mult=0.0),
#
#             # 2. 冻结 Text Encoder (CLIP)
#             'text_model': dict(lr_mult=0.0),
#
#             # 3. 冻结 Neck (PAFPN)
#             'neck': dict(lr_mult=0.0),
#
#             # 4. 冻结 Logit Scale (温度系数)
#             'logit_scale': dict(lr_mult=0.0),
#
#             # 5. Head 部分学习率乘以0.1
#             'bbox_head': dict(lr_mult=0.0),
#
#             # 6. [核心] 唯独解冻 SAVPE
#             # 这里的 key 必须写得比 'bbox_head' 更长、更具体，以触发最长匹配原则
#             # 参数的全名通常是: bbox_head.head_module.savpe.xxx
#             'bbox_head.head_module.savpe': dict(lr_mult=1.0, decay_mult=1.0),
#             'bbox_head.head_module.opr_fusion.visual_adapter': dict(lr_mult=1.0, decay_mult=1.0),
#         }
#     ),
#     constructor='YOLOWv5OptimizerConstructor'
# )


#第二阶段，融合模块1.0，图像提示0.1，head冻住
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),

    # [核心修改] 参数级精细化配置
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,

        # custom_keys 字典：键是参数名的一部分，值是配置
        custom_keys={
            # 1. 冻结 Backbone (包含 Image Encoder)
            'backbone': dict(lr_mult=0.0),

            # 2. 冻结 Text Encoder (CLIP)
            'text_model': dict(lr_mult=0.0),

            # 3. 冻结 Neck (PAFPN)
            'neck': dict(lr_mult=0.0),

            # 4. 冻结 Logit Scale (温度系数)
            'logit_scale': dict(lr_mult=0.0),

            # 5. Head 部分学习率乘以0.1
            'bbox_head': dict(lr_mult=0.0),

            # 6. [核心] 唯独解冻 SAVPE
            # 这里的 key 必须写得比 'bbox_head' 更长、更具体，以触发最长匹配原则
            # 参数的全名通常是: bbox_head.head_module.savpe.xxx
            'bbox_head.head_module.savpe': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head.head_module.opr_fusion': dict(lr_mult=1.0, decay_mult=1.0),
            'bbox_head.head_module.opr_fusion.visual_adapter': dict(lr_mult=0.1, decay_mult=1.0),
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)





#load_from = 'work_dirs/train_deformable_nofuse_val_minival/20251203_193453/epoch_2.pth'
#load_from = 'official_pretraind_models/yolo-world-l-640.pth'
#load_from = 'work_dirs/finetune_deformable_contrast_fuse_fromofficialyoloworld_val_minival/epoch_2.pth'
#load_from = '/data/codes/WangShuo/py_project/YOLO-World-research/YOLO-World/work_dirs/finetune_contrast_fusev2_fromofficialyoloworld_val_minival/20251216_213704/epoch_1.pth'
load_from = 'work_dirs/finetune_deformablev2_only_load_officialmodel_lr2e-4_close_mosaic_5epoch/20251224_000816/epoch_3.pth'


model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)