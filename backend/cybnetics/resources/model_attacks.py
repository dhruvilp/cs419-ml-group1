import os
import torch
import cv2

from bson import ObjectId
from pymongo import ReturnDocument

from cybnetics import utils
from .cnn_classes import CIFARNet, MNISTNet
from .db import models_coll
from . import models

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

class InvalidAttack(Exception):
    def __str__(self):
        return 'one of the one million things went wrong during the attack'

def convert_tensor(filename, model_type):
    """ Converts the image file to a tensor """
    try:
        if model_type == 'mnist':
            # Load the attack image in greyscale with the shape (C * H * W)
            img = cv2.imread(filename, 0).reshape([1,28,28])
            img_tensor = torch.from_numpy(img).reshape([1,1,28,28])
            return img_tensor.float()
        elif model_type == 'cifar':
            # Load the attack image in rgb with the shape (C * H * W)
            img = cv2.imread(filename, 1).transpose((2, 0, 1))
            img_tensor = torch.from_numpy(img).reshape([1,3,32,32])
            return img_tensor.float()
    except:
        raise InvalidImage()

def prime_neural_netowrk(model_path, model_type):
    device = torch.device('cpu')
    if model_type == 'mnist':
        net = MNISTNet().to(device)
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        return net.eval()
    elif model_type == 'cifar':
        net = CIFARNet().to(device)
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        return net.eval()

def attack_model(model_path, model_type, input_tensor, label):
    net = prime_neural_netowrk(model_path, model_type)
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
    try:
        image_path = utils.get_path('a_' + str(model_id) + '_' + str(ObjectId()))
        attack_image.save(image_path)
        db_models = models_coll()
        is_model = db_models.find_one({'_id': model_id})
        model_type = is_model['model_type']
        input_tensor = convert_tensor(image_path, model_type)
        model_path = utils.model_path(model_id)
        success = attack_model(model_path, model_type, input_tensor, label)
        return success
    except:
        raise InvalidAttack()
    finally:
        os.remove(image_path)

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
