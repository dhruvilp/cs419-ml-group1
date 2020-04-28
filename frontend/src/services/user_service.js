import { authService } from '../services/auth_service.js';

const BASE_URL = 'http://127.0.0.1:5000';

export const userService = {
    getLeaderboard,
    getListOfModels,
    getModelDescription,
    downloadMlModel,
    downloadDataset,
    deleteModel,
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
async function getListOfModels() {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders
    };
    const response = await fetch(`${BASE_URL}/models`, requestOptions);
    const listOfModels = await handleResponse(response);
    return listOfModels;
}

//================= GET MODEL DESCRIPTION ==================
async function getModelDescription(id) {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders
    };
    const response = await fetch(`${BASE_URL}/models/${id}`, requestOptions);
    const modelDescription = await handleResponse(response);
    return modelDescription;
}

//=============== DOWNLOAD ML MODEL ==================
async function downloadMlModel(id) {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders
    };
    const response = await fetch(`${BASE_URL}/models/${id}/model`, requestOptions);
    const downloadedMlModel = await handleResponse(response);
    return downloadedMlModel;
}

//=============== DOWNLOAD DATASET ==================
async function downloadDataset(id) {
    const requestOptions = {
        method: 'GET',
        headers: kHeaders
    };
    const response = await fetch(`${BASE_URL}/models/${id}/dataset`, requestOptions);
    const downloadedDataset = await handleResponse(response);
    return downloadedDataset;
}

//=============== DELETE MODEL ==================
async function deleteModel(id) {
    const requestOptions = {
        method: 'DELETE',
        headers: kHeaders
    };
    const response = await fetch(`${BASE_URL}/models/${id}`, requestOptions);
    const deletedModel = await handleResponse(response);
    return deletedModel;
}

//=============== ADMIN PUBLISH NEW MODEL ==================
async function adminPublishNewModel(name, description, type, mode) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: JSON.stringify({ 
            "name": name,
            "description": description,
            "model_type": type,
            "attack_mode": mode
        })
    };
    const response = await fetch(`${BASE_URL}/models`, requestOptions);
    const publishedNewModel = await handleResponse(response);
    return publishedNewModel;
}

//=============== UPLOAD TRAINED MODEL ==================
async function uploadTrainedModel(id, formData) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: formData
    };
    const response = await fetch(`${BASE_URL}/models/${id}/model`, requestOptions);
    const uploadedTrainedModel = await handleResponse(response);
    return uploadedTrainedModel;
}

//=============== UPLOAD DATSET ==================
async function uploadDataset(id, formData) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: formData
    };
    const response = await fetch(`${BASE_URL}/models/${id}/dataset`, requestOptions);
    const uploadedDataset = await handleResponse(response);
    return uploadedDataset;
}

//=============== UPLOAD DATSET ==================
async function uploadFileToAttack(id, formData, label) {
    const requestOptions = {
        method: 'POST',
        headers: kHeaders,
        body: formData
    };
    const response = await fetch(`${BASE_URL}/models/${id}/attack?label=${label}`, requestOptions);
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
