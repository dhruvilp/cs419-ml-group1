import os
import torch
import cv2

from bson import ObjectId
from pymongo import ReturnDocument

from . import models
from .db import models_coll
from cybnetics import utils
from cybnetics.model_builder import make_model_class

SCORE_MAP = {
    'white': 1,
    'gray': 5,
    'black': 10,
}

PLACE_MAP = {
    1: 'gold',
    2: 'silver',
    3: 'bronze'
}

class InvalidImage(Exception):
    def __str__(self):
        return 'the image was invalid'

class InvalidAttack(ValueError):
    pass

def convert_tensor(filename, color):
    """
    Generalize convert_tensor (there are tensor functions that can make this easier
    can't think of them right now; this works though)

        args:
            filename: string
            color: bool
    """
    flag = cv2.IMREAD_COLOR
    if not color:
        flag = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(filename, flag)

    if not color:
        init_reshape = img.reshape([1, 1, img.shape[0], img.shape[1]])
        img_tensor = torch.from_numpy(init_reshape)
        return img_tensor.float()
    else:
        init_reshape = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(init_reshape).reshape([1,3, img.shape[0], img.shape[1]])
        return img_tensor.float()


def prime_neural_network(model_path, layers, pools):
    klass = make_model_class(layers, pools)
    device = torch.device('cpu')
    net = klass().to(device)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    return net.eval()

def attack_model(net, input_tensor, label):
    output = net(input_tensor)
    predicted_label = output.max(1, keepdim=True)[1]
    if predicted_label.item() == int(label):
        # the attack was not a success
        return False
    else:
        # attack was a success
        return True

def simulate_attack(model_id, label, attack_image, user):
    """ One Million things can go wrong in this method """
    # if we get an exception then success = False
    success = False
    # make paths
    image_path = utils.get_path('a_' + str(model_id) + '_' + str(ObjectId()))
    model_path = utils.model_filename(model_id)
    # temp save the image
    attack_image.save(image_path)
    try:
        # get model from db
        db_models = models_coll()
        model = db_models.find_one({'_id': model_id})
        # make the net to classify the input
        net = prime_neural_network(model_path, model['layers'], model['pools'])
        # convert input into tensor
        input_tensor = convert_tensor(image_path, model['color'])
        # FINALY do the damn attack
        success = attack_model(net, input_tensor, label)
    except Exception as e:
        raise InvalidAttack(str(e))
    os.remove(image_path)
    return success

def set_place(attack_id, place_name):
    models_c = models_coll()
    models_c.update_one({
        'attacks._id': attack_id
    }, {
        '$set': {'attacks.$.place': place_name} # assign the attack its place
    })

def save_attack(model_id, label, user, success):
    models_c = models_coll()
    attack_id = ObjectId()
    points = 0
    if success:
        model = models.find_one(model_id)
        points = SCORE_MAP[model['attack_mode']]
    # add attack to model document in db.
    result = models_c.find_one_and_update({
        '_id': model_id
    }, {
        '$push': {
            'attacks': {
                '_id': attack_id,
                'label': label,
                'user': user,
                'success': success,
                'points': points
            }
        }
    }, return_document=ReturnDocument.AFTER)
    attack = result['attacks'][-1]
    # see if inserted document needs to be assigned gold, silver, ect
    # lmao this gets slower with more and more attacks
    place = len(list(filter(lambda attack: attack['success'], result['attacks'])))
    if place <= 3 and success:
        attack['place'] = PLACE_MAP[place]
        set_place(attack_id, PLACE_MAP[place])
    return attack
