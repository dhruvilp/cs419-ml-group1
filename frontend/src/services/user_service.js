import { authService } from '../services/auth_service.js';

const BASE_URL = 'http://127.0.0.1:5000';

export const userService = {
    getLeaderboard,
    getListOfModels,
    getModelDescription,
    downloadMlModel,
    downloadDataset,
    adminPublishNewModel,
    uploadTrainedModel,
    uploadDataset,
    uploadFileToAttack
};

var authToken = '';
const getStoredCred = JSON.parse(localStorage.getItem('user'));
if(getStoredCred !== null){
    authToken = getStoredCred['token'];
}

const kHeaders = new Headers({
    'Content-Type': 'application/json',
    'Cybnetics-Token': authToken 
  });

//=================== GET SCORECARD DATA ====================
async function getLeaderboard() {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders
    };
    const response = await fetch(`${BASE_URL}/scoreboard`, requestOptions);
    const scoreboardData = await handleResponse(response);
    return scoreboardData;
}

//================= GET LIST OF MODELS ==================
async function getListOfModels(username) {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders
    };
    const response = await fetch(`${BASE_URL}/models?user=${username}`, requestOptions);
    const listOfModels = await handleResponse(response);
    return listOfModels;
}

//================= GET MODEL DESCRIPTION ==================
async function getModelDescription(id) {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders,
        body: JSON.stringify({ 
            "token" : authToken
        })
    };
    const response = await fetch(`${BASE_URL}/models/${id}`, requestOptions);
    const modelDescription = await handleResponse(response);
    return modelDescription;
}

//=============== DOWNLOAD ML MODEL ==================
async function downloadMlModel(id) {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders,
        body: JSON.stringify({ 
            "token" : authToken
        })
    };
    const response = await fetch(`${BASE_URL}/models/${id}/model`, requestOptions);
    const downloadedMlModel = await handleResponse(response);
    return downloadedMlModel;
}

//=============== DOWNLOAD DATASET ==================
async function downloadDataset(id) {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders,
        body: JSON.stringify({ 
            "token" : authToken
        })
    };
    const response = await fetch(`${BASE_URL}/models/${id}/dataset`, requestOptions);
    const downloadedDataset = await handleResponse(response);
    return downloadedDataset;
}

//=============== ADMIN PUBLISH NEW MODEL ==================
async function adminPublishNewModel(name, description, type, mode) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: JSON.stringify({ 
            "token" : authToken,
            "name": name,
            "description": description,
            "model_type": type,
            "attack_mode": mode
        })
    };
    const response = await fetch(`${BASE_URL}/models`, requestOptions);
    const downloadedDataset = await handleResponse(response);
    return downloadedDataset;
}

//=============== UPLOAD TRAINED MODEL ==================
async function uploadTrainedModel(id) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: JSON.stringify({ 
            "token" : authToken
            // static_dict / ML Model YET to be implemented
        })
    };
    const response = await fetch(`${BASE_URL}/models/${id}/model`, requestOptions);
    const uploadedTrainedModel = await handleResponse(response);
    return uploadedTrainedModel;
}

//=============== UPLOAD DATSET ==================
async function uploadDataset(id) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: JSON.stringify({ 
            "token" : authToken
            // dataset file required, YET to be implemented
        })
    };
    const response = await fetch(`${BASE_URL}/models/${id}/dataset`, requestOptions);
    const uploadedDataset = await handleResponse(response);
    return uploadedDataset;
}

//=============== UPLOAD DATSET ==================
async function uploadFileToAttack(id) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: JSON.stringify({ 
            "token" : authToken,
            "label": "ORIGINAL_LABEL" // YET TO CHANGE
            // model or image file required, YET to be implemented
        })
    };
    const response = await fetch(`${BASE_URL}/models/${id}/attack`, requestOptions);
    const uploadedAttackFile = await handleResponse(response);
    return uploadedAttackFile;
}


//==================================================
//                 MISC FUNCTIONS
//==================================================
function handleResponse(response) {
    return response.text().then(text => {
        const data = (response.ok) && text && JSON.parse(text);
        if (!response.ok) {
            if (response.status === 401) {
                // auto logout if 401 response returned from api
                authService.logout();
                // location.reload(true);
            }
            const error = (data && data.message) || response.statusText;
            return Promise.reject(error);
        }
        return data;
    });
}
