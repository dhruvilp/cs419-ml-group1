export const userService = {
    getLeaderboard
};

const BASE_URL = 'http://127.0.0.1:5000';

function getLeaderboard() {

}

function handleResponse(response) {
    return response.text().then(text => {
        const data = (response.ok) && text && JSON.parse(text);
        if (!response.ok) {
            if (response.status === 401) {
                // auto logout if 401 response returned from api
                logout();
                // location.reload(true);
            }
            const error = (data && data.message) || response.statusText;
            return Promise.reject(error);
        }
        return data;
    });
}