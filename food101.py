import tensorflow as tf
import cv2
import numpy as np


def fastFood101Modellabels( ):
    # dataset Model Labels
    fastfood101Labels = [
                    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding',
                    'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate',
                    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich',
                    'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
                    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
                    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
                    'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
                    'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
                    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
                    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
                    'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
                ]

    return fastfood101Labels

# instantiate the model definition
def fastFood101Model():
        # base model
        tf.keras.backend.clear_session()
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.EfficientNetB0( include_top = False )
        base_model.trainable = False
        class_names_length = len( fastFood101Modellabels() )
        # create functional layers of model
        inputs = tf.keras.layers.Input(shape = input_shape, name="input_layer")
        # We will be using Efficientnet which has built in scaling (1/255.0)
        # x = preprocessing.Rescaling(1/255.0)(x) #We won't be needing this layer for Efficientnet
        x = base_model(inputs, training = False) # make sure the layers should be in inference mode - NOT TRAINING
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense( class_names_length )(x)
        output = tf.keras.layers.Activation('softmax', dtype = tf.float32, name="softmax_float32")(x)
        model = tf.keras.Model(inputs, output)
        # compile the model
        model.compile( loss="sparse_categorical_crossentropy", #sparse_categorical_crossentropy for labels that are *not* one-hot
                optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
                metrics=["accuracy"])
        
        return model
